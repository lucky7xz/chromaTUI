use std::collections::VecDeque;
use std::process::Command;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::SampleFormat;

/// Enough for the largest FFT window (16384) with headroom.
const CAPACITY: usize = 1 << 15;

/// What the live capture stream is currently pointed at.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum InputSource {
    Mic,
    Internal,
}

impl InputSource {
    pub fn label(self) -> &'static str {
        match self {
            InputSource::Mic => "microphone",
            InputSource::Internal => "internal audio",
        }
    }

    /// Short form for the always-visible panel footer.
    pub fn short(self) -> &'static str {
        match self {
            InputSource::Mic => "mic",
            InputSource::Internal => "internal",
        }
    }
}

pub struct AudioInput {
    pub sample_rate: f32,
    pub device_name: String,
    pub source: InputSource,
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
            source: InputSource::Mic,
            buffer,
            _stream: stream,
        })
    }

    /// Flip between the mic and the default sink's monitor.
    pub fn toggle_source(&mut self) -> Result<()> {
        let next = match self.source {
            InputSource::Mic => InputSource::Internal,
            InputSource::Internal => InputSource::Mic,
        };
        self.set_source(next)
    }

    /// Reroute the *live* capture stream to `src` at the PipeWire/Pulse level.
    ///
    /// ALSA cannot name monitor sources, so re-opening the cpal stream could
    /// never reach internal audio; moving the existing source-output can.
    /// On failure `self.source` is left untouched.
    pub fn set_source(&mut self, src: InputSource) -> Result<()> {
        let target = match src {
            InputSource::Mic => pactl(&["get-default-source"])?,
            // The default sink's monitor, so this follows speakers/headphones.
            InputSource::Internal => format!("{}.monitor", pactl(&["get-default-sink"])?),
        };
        let id = our_source_output_id()?;
        pactl(&["move-source-output", &id.to_string(), &target])?;
        self.source = src;
        Ok(())
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

/// Run pactl and return trimmed stdout.
fn pactl(args: &[&str]) -> Result<String> {
    let out = Command::new("pactl")
        .args(args)
        .output()
        .context("pactl not found — input switching needs PipeWire or PulseAudio")?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(anyhow!("pactl {}: {}", args[0], err.trim()));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// Find the id of *our* capture stream by matching the process id that
/// PipeWire records on each source-output.
fn our_source_output_id() -> Result<u32> {
    let listing = pactl(&["list", "source-outputs"])?;
    let me = std::process::id().to_string();
    let mut current = None;
    for line in listing.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Source Output #") {
            current = rest.trim().parse::<u32>().ok();
        } else if let Some(pid) = line.strip_prefix("application.process.id = ") {
            if pid.trim().trim_matches('"') == me {
                if let Some(id) = current {
                    return Ok(id);
                }
            }
        }
    }
    Err(anyhow!("capture stream not registered with the audio server"))
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
