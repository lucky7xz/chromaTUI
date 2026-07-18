use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;

/// Enough for the largest FFT window (16384) with headroom.
const CAPACITY: usize = 1 << 15;

pub struct AudioInput {
    pub sample_rate: f32,
    pub device_name: String,
    buffer: Arc<Mutex<VecDeque<f32>>>,
    _stream: cpal::Stream,
}

impl AudioInput {
    /// Open the default input device (mic) mono-mixed into a ring buffer.
    pub fn start() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("no audio input device found — is a microphone connected?")?;
        let device_name = device.name().unwrap_or_else(|_| "unknown".into());
        let config = device
            .default_input_config()
            .context("input device has no default config")?;
        let sample_rate = config.sample_rate().0 as f32;
        let channels = config.channels() as usize;

        let buffer = Arc::new(Mutex::new(VecDeque::with_capacity(CAPACITY)));
        let err_fn = |_e: cpal::StreamError| {};

        macro_rules! stream {
            ($t:ty, $conv:expr) => {{
                let buf = buffer.clone();
                device.build_input_stream(
                    &config.into(),
                    move |data: &[$t], _: &_| push_mono(&buf, data, channels, $conv),
                    err_fn,
                    None,
                )?
            }};
        }

        let stream = match config.sample_format() {
            SampleFormat::F32 => stream!(f32, |s: f32| s),
            SampleFormat::I16 => stream!(i16, |s: i16| s as f32 / i16::MAX as f32),
            SampleFormat::U16 => stream!(u16, |s: u16| (s as f32 - 32768.0) / 32768.0),
            fmt => anyhow::bail!("unsupported sample format {fmt:?}"),
        };
        stream.play().context("failed to start audio stream")?;

        Ok(AudioInput {
            sample_rate,
            device_name,
            buffer,
            _stream: stream,
        })
    }

    /// Copy the most recent `out.len()` samples, zero-padding the front
    /// when not enough audio has arrived yet.
    pub fn copy_latest(&self, out: &mut [f32]) {
        let buf = self.buffer.lock().unwrap();
        let n = out.len();
        let have = buf.len().min(n);
        let pad = n - have;
        out[..pad].fill(0.0);
        for (dst, &s) in out[pad..].iter_mut().zip(buf.iter().skip(buf.len() - have)) {
            *dst = s;
        }
    }
}

fn push_mono<T: Copy>(
    buffer: &Arc<Mutex<VecDeque<f32>>>,
    data: &[T],
    channels: usize,
    conv: impl Fn(T) -> f32,
) {
    let mut buf = buffer.lock().unwrap();
    for frame in data.chunks(channels.max(1)) {
        let mono = frame.iter().map(|&s| conv(s)).sum::<f32>() / frame.len() as f32;
        buf.push_back(mono);
    }
    let excess = buf.len().saturating_sub(CAPACITY);
    if excess > 0 {
        buf.drain(..excess);
    }
}
