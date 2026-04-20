use super::{VadFrame, VoiceActivityDetector};
use anyhow::Result;

/// Pre-allocated circular frame buffer that avoids per-frame heap allocations.
/// Stores up to `capacity` frames of `frame_size` samples in a flat Vec.
struct FrameRing {
    data: Vec<f32>,
    frame_size: usize,
    capacity: usize,
    head: usize,
    count: usize,
}

impl FrameRing {
    fn new(capacity: usize, frame_size: usize) -> Self {
        Self {
            data: vec![0.0; capacity * frame_size],
            frame_size,
            capacity,
            head: 0,
            count: 0,
        }
    }

    fn push(&mut self, frame: &[f32]) {
        let write_idx = (self.head + self.count) % self.capacity;
        let offset = write_idx * self.frame_size;
        self.data[offset..offset + self.frame_size].copy_from_slice(frame);

        if self.count < self.capacity {
            self.count += 1;
        } else {
            self.head = (self.head + 1) % self.capacity;
        }
    }

    /// Drain all frames into a flat Vec, oldest first. Resets the ring.
    fn drain_to_vec(&mut self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.count * self.frame_size);
        for i in 0..self.count {
            let idx = (self.head + i) % self.capacity;
            let offset = idx * self.frame_size;
            out.extend_from_slice(&self.data[offset..offset + self.frame_size]);
        }
        self.head = 0;
        self.count = 0;
        out
    }

    fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
    }

    /// Remove the most recently pushed frame.
    fn pop_back(&mut self) {
        if self.count > 0 {
            self.count -= 1;
        }
    }
}

pub struct SmoothedVad {
    inner_vad: Box<dyn VoiceActivityDetector>,
    prefill_frames: usize,
    hangover_frames: usize,
    onset_frames: usize,

    frame_ring: FrameRing,
    hangover_counter: usize,
    onset_counter: usize,
    in_speech: bool,

    temp_out: Vec<f32>,
}

impl SmoothedVad {
    pub fn new(
        inner_vad: Box<dyn VoiceActivityDetector>,
        prefill_frames: usize,
        hangover_frames: usize,
        onset_frames: usize,
    ) -> Self {
        let frame_size = 480; // SILERO_FRAME_SAMPLES (30ms @ 16kHz)
        let ring_capacity = if prefill_frames > 0 {
            prefill_frames + 1
        } else {
            0
        };

        Self {
            inner_vad,
            prefill_frames,
            hangover_frames,
            onset_frames,
            frame_ring: FrameRing::new(ring_capacity, frame_size),
            hangover_counter: 0,
            onset_counter: 0,
            in_speech: false,
            temp_out: Vec::with_capacity(ring_capacity * frame_size),
        }
    }
}

impl VoiceActivityDetector for SmoothedVad {
    fn push_frame<'a>(&'a mut self, frame: &'a [f32]) -> Result<VadFrame<'a>> {
        if self.prefill_frames > 0 {
            self.frame_ring.push(frame);
        }

        let is_voice = self.inner_vad.is_voice(frame)?;

        match (self.in_speech, is_voice) {
            (false, true) => {
                self.onset_counter += 1;
                if self.onset_counter >= self.onset_frames {
                    self.in_speech = true;
                    self.hangover_counter = self.hangover_frames;
                    self.onset_counter = 0;

                    // Pop the current frame — caller already has it
                    self.frame_ring.pop_back();
                    self.temp_out = self.frame_ring.drain_to_vec();
                    self.temp_out.extend_from_slice(frame);
                    Ok(VadFrame::Speech(&self.temp_out))
                } else {
                    Ok(VadFrame::Noise)
                }
            }

            (true, true) => {
                self.hangover_counter = self.hangover_frames;
                Ok(VadFrame::Speech(frame))
            }

            (true, false) => {
                if self.hangover_counter > 0 {
                    self.hangover_counter -= 1;
                    Ok(VadFrame::Speech(frame))
                } else {
                    self.in_speech = false;
                    Ok(VadFrame::Noise)
                }
            }

            (false, false) => {
                self.onset_counter = 0;
                Ok(VadFrame::Noise)
            }
        }
    }

    fn reset(&mut self) {
        self.frame_ring.clear();
        self.hangover_counter = 0;
        self.onset_counter = 0;
        self.in_speech = false;
        self.temp_out.clear();
    }
}
