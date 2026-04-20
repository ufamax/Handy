use anyhow::Result;
use ndarray::{Array1, Array3, ArrayView2};
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

use super::{VadFrame, VoiceActivityDetector};
use crate::audio_toolkit::constants;

const SILERO_FRAME_MS: u32 = 30;
const SILERO_FRAME_SAMPLES: usize =
    (constants::WHISPER_SAMPLE_RATE * SILERO_FRAME_MS / 1000) as usize;

/// Silero VAD v4 — CPU-only ONNX session optimized for low latency.
///
/// The Silero LSTM model is tiny and runs 33× per second.  GPU execution
/// adds kernel-launch and PCIe-transfer overhead that exceeds the compute
/// savings, and cuDNN's RNN kernels can fail on some driver combinations.
/// Keeping VAD on CPU with a single thread is both faster and more robust.
///
/// Pre-allocates LSTM state tensors to avoid per-frame heap allocations
/// (zero-copy via `TensorRef`).
pub struct SileroVad {
    session: Session,
    h: Array3<f32>,
    c: Array3<f32>,
    sr: Array1<i64>,
    threshold: f32,
}

impl SileroVad {
    pub fn new<P: AsRef<Path>>(model_path: P, threshold: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&threshold) {
            anyhow::bail!("threshold must be between 0.0 and 1.0");
        }

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("VAD session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("VAD optimization level: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("VAD intra threads: {e}"))?
            .with_inter_threads(1)
            .map_err(|e| anyhow::anyhow!("VAD inter threads: {e}"))?
            .commit_from_file(model_path.as_ref())
            .map_err(|e| anyhow::anyhow!("Failed to load VAD model: {e}"))?;

        Ok(Self {
            session,
            h: Array3::zeros((2, 1, 64)),
            c: Array3::zeros((2, 1, 64)),
            sr: Array1::from_vec(vec![16000i64]),
            threshold,
        })
    }

    fn speech_probability(&mut self, frame: &[f32]) -> Result<f32> {
        let input = ArrayView2::from_shape((1, SILERO_FRAME_SAMPLES), frame)
            .map_err(|e| anyhow::anyhow!("VAD input shape error: {e}"))?;

        let t_input = TensorRef::from_array_view(input.into_dyn())
            .map_err(|e| anyhow::anyhow!("tensor input: {e}"))?;
        let t_sr = TensorRef::from_array_view(self.sr.view().into_dyn())
            .map_err(|e| anyhow::anyhow!("tensor sr: {e}"))?;
        let t_h = TensorRef::from_array_view(self.h.view().into_dyn())
            .map_err(|e| anyhow::anyhow!("tensor h: {e}"))?;
        let t_c = TensorRef::from_array_view(self.c.view().into_dyn())
            .map_err(|e| anyhow::anyhow!("tensor c: {e}"))?;

        let outputs = self
            .session
            .run(inputs![
                "input" => t_input,
                "sr" => t_sr,
                "h" => t_h,
                "c" => t_c,
            ])
            .map_err(|e| anyhow::anyhow!("VAD inference error: {e}"))?;

        let hn = outputs
            .get("hn")
            .ok_or_else(|| anyhow::anyhow!("missing VAD output: hn"))?
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("extract hn: {e}"))?;
        self.h = hn
            .to_owned()
            .into_shape_with_order((2, 1, 64))
            .map_err(|e| anyhow::anyhow!("reshape hn: {e}"))?;

        let cn = outputs
            .get("cn")
            .ok_or_else(|| anyhow::anyhow!("missing VAD output: cn"))?
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("extract cn: {e}"))?;
        self.c = cn
            .to_owned()
            .into_shape_with_order((2, 1, 64))
            .map_err(|e| anyhow::anyhow!("reshape cn: {e}"))?;

        let output = outputs
            .get("output")
            .ok_or_else(|| anyhow::anyhow!("missing VAD output: output"))?
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("extract output: {e}"))?;

        Ok(output[[0, 0]])
    }
}

impl VoiceActivityDetector for SileroVad {
    fn push_frame<'a>(&'a mut self, frame: &'a [f32]) -> Result<VadFrame<'a>> {
        if frame.len() != SILERO_FRAME_SAMPLES {
            anyhow::bail!(
                "expected {SILERO_FRAME_SAMPLES} samples, got {}",
                frame.len()
            );
        }

        let prob = self.speech_probability(frame)?;

        if prob > self.threshold {
            Ok(VadFrame::Speech(frame))
        } else {
            Ok(VadFrame::Noise)
        }
    }

    fn reset(&mut self) {
        self.h.fill(0.0);
        self.c.fill(0.0);
    }
}
