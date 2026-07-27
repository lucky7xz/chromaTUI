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

        // Don't assume mic: the server may have restored last session's route.
        let source = detect_source().unwrap_or(InputSource::Mic);

        Ok(AudioInput {
            sample_rate,
            device_name,
            source,
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
///
/// `LC_ALL=C` is not optional: `pactl list` field names ("Source Output #",
/// "Source:") go through gettext, so on a localised desktop the listing we
/// parse below comes back translated and matches nothing.
fn pactl(args: &[&str]) -> Result<String> {
    let out = Command::new("pactl")
        .env("LC_ALL", "C")
        .env("LANGUAGE", "")
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
    let pid = std::process::id().to_string();
    match parse_our_stream(&listing, &pid) {
        Some((id, _)) => Ok(id),
        None => {
            dump_diagnostics(&listing);
            Err(not_registered(&listing, &pid))
        }
    }
}

/// One `Source Output #…` block, reduced to what identifies it.
struct SourceOutput {
    id: u32,
    source: Option<u32>,
    pid: Option<String>,
    app: Option<String>,
}

/// Split `pactl list source-outputs` into blocks. Properties we don't need are
/// ignored, and a block with an unparseable header is dropped.
fn parse_source_outputs(listing: &str) -> Vec<SourceOutput> {
    let mut outs: Vec<SourceOutput> = Vec::new();
    for line in listing.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("Source Output #") {
            if let Ok(id) = rest.trim().parse::<u32>() {
                outs.push(SourceOutput { id, source: None, pid: None, app: None });
            }
            continue;
        }
        let Some(cur) = outs.last_mut() else { continue };
        let value = |s: &str| s.trim().trim_matches('"').to_string();
        if let Some(rest) = line.strip_prefix("Source:") {
            cur.source = rest.trim().parse::<u32>().ok();
        } else if let Some(rest) = line.strip_prefix("application.process.id = ") {
            cur.pid = Some(value(rest));
        } else if let Some(rest) = line.strip_prefix("application.name = ") {
            cur.app = Some(value(rest));
        }
    }
    outs
}

/// Locate our own capture stream; returns (source-output id, source index).
///
/// Process id is the reliable key when the server records one. Some
/// ALSA→PipeWire paths don't, so we fall back to the block named after us —
/// but only if there is exactly one, since guessing would move a stranger's
/// stream.
fn parse_our_stream(listing: &str, pid: &str) -> Option<(u32, u32)> {
    let outs = parse_source_outputs(listing);
    let by_pid = outs.iter().find(|o| o.pid.as_deref() == Some(pid));
    let ours = by_pid.or_else(|| {
        let mut named = outs
            .iter()
            .filter(|o| o.app.as_deref().is_some_and(is_us) && o.pid.is_none());
        named.next().filter(|_| named.next().is_none())
    })?;
    Some((ours.id, ours.source?))
}

/// PipeWire/Pulse label our stream "ALSA plug-in [chromatui]"; be lenient
/// about the wrapper text.
fn is_us(app: &str) -> bool {
    app.to_ascii_lowercase().contains("chromatui")
}

/// The one failure users actually hit, worded so a screenshot tells us which
/// cause it is: nothing listed at all (capture opened outside the sound
/// server, so rerouting is impossible) versus listed-but-not-ours (our
/// identifier is wrong).
fn not_registered(listing: &str, pid: &str) -> anyhow::Error {
    let outs = parse_source_outputs(listing);
    if outs.is_empty() {
        return anyhow!(
            "no capture streams listed by pactl — mic opened outside PipeWire/PulseAudio"
        );
    }
    let seen: Vec<String> = outs
        .iter()
        .map(|o| match (&o.pid, &o.app) {
            (Some(p), _) => format!("pid {p}"),
            (None, Some(a)) => format!("\"{a}\" (no pid)"),
            (None, None) => "unnamed".into(),
        })
        .collect();
    anyhow!(
        "pid {pid} not among {} listed streams: {}",
        outs.len(),
        seen.join(", ")
    )
}

/// Write the raw listing next to `pactl info` so the failure can be diagnosed
/// from one run on a machine we don't have. Best effort — never fails a switch.
fn dump_diagnostics(listing: &str) {
    let path = std::env::temp_dir().join("chromatui-audio-debug.log");
    let info = pactl(&["info"]).unwrap_or_else(|e| format!("<pactl info failed: {e}>"));
    let sources = pactl(&["list", "short", "sources"]).unwrap_or_default();
    let _ = std::fs::write(
        &path,
        format!(
            "chromatui pid {}\n\n== pactl info ==\n{info}\n\n\
             == list source-outputs ==\n{listing}\n\n== list short sources ==\n{sources}\n",
            std::process::id()
        ),
    );
}

/// Parse `pactl list short sources` (tab-separated `idx\tname\t…`) for the
/// name of source `idx`.
fn source_name_for_index(short_listing: &str, idx: u32) -> Option<&str> {
    short_listing.lines().find_map(|line| {
        let mut cols = line.split('\t');
        (cols.next()?.trim().parse::<u32>().ok()? == idx).then(|| cols.next())?
    })
}

/// Where did the server actually route our stream? WirePlumber's
/// restore-stream re-pins new streams by application name, so the *last
/// session's* choice wins over any assumption — quit on internal and the
/// next launch starts on internal. `None` = can't tell (no pactl, or the
/// stream never registered); the caller falls back to Mic.
fn detect_source() -> Option<InputSource> {
    let me = std::process::id().to_string();
    // Stream registration is async relative to `stream.play()` returning.
    let mut found = None;
    for _ in 0..20 {
        let listing = pactl(&["list", "source-outputs"]).ok()?;
        if let Some(hit) = parse_our_stream(&listing, &me) {
            found = Some(hit);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    found?;
    // Re-read after a beat: restore-stream may move the node right after it
    // appears, and we want where it settled, not where it was born.
    std::thread::sleep(std::time::Duration::from_millis(50));
    let listing = pactl(&["list", "source-outputs"]).ok()?;
    let (_, src_idx) = parse_our_stream(&listing, &me)?;
    let sources = pactl(&["list", "short", "sources"]).ok()?;
    let name = source_name_for_index(&sources, src_idx)?;
    Some(if name.ends_with(".monitor") {
        InputSource::Internal
    } else {
        InputSource::Mic
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Real `pactl list source-outputs` shape from this machine: two blocks,
    /// ours second.
    const LISTING: &str = "\
Source Output #1346
\tDriver: protocol-native.c
\tSource: 51
\tSample Specification: s16le 2ch 44100Hz
\tProperties:
\t\tapplication.name = \"ALSA plug-in [aplay]\"
\t\tapplication.process.id = \"11111\"
Source Output #1398
\tDriver: protocol-native.c
\tSource: 53
\tSample Specification: s24-32le 2ch 48000Hz
\tProperties:
\t\tapplication.name = \"ALSA plug-in [chromatui]\"
\t\tapplication.process.id = \"16844\"
";

    /// Same shape, but the server recorded no `application.process.id` for
    /// our stream — the case pid matching cannot solve.
    const LISTING_NO_PID: &str = "\
Source Output #1346
\tDriver: protocol-native.c
\tSource: 51
\tProperties:
\t\tapplication.name = \"ALSA plug-in [aplay]\"
\t\tapplication.process.id = \"11111\"
Source Output #1398
\tDriver: protocol-native.c
\tSource: 53
\tProperties:
\t\tapplication.name = \"ALSA plug-in [chromatui]\"
";

    const SOURCES: &str = "\
51\talsa_output.pci-0000_00_1f.3.HiFi__hw_sofhdadsp__sink.monitor\tPipeWire\ts24-32le 2ch 48000Hz\tRUNNING
53\talsa_input.pci-0000_00_1f.3.HiFi__hw_sofhdadsp_6__source\tPipeWire\ts32le 2ch 48000Hz\tIDLE
";

    #[test]
    fn finds_our_block_among_several() {
        assert_eq!(parse_our_stream(LISTING, "16844"), Some((1398, 53)));
        assert_eq!(parse_our_stream(LISTING, "11111"), Some((1346, 51)));
        assert_eq!(parse_our_stream(LISTING, "99999"), None);
    }

    #[test]
    fn falls_back_to_app_name_when_pid_is_missing() {
        // pid 99999 is not in the listing, but exactly one block is ours by name.
        assert_eq!(parse_our_stream(LISTING_NO_PID, "99999"), Some((1398, 53)));
    }

    #[test]
    fn refuses_to_guess_between_two_same_named_blocks() {
        let two = LISTING_NO_PID.to_string()
            + "Source Output #1400\n\tSource: 54\n\tProperties:\n\
               \t\tapplication.name = \"ALSA plug-in [chromatui]\"\n";
        assert_eq!(parse_our_stream(&two, "99999"), None);
    }

    #[test]
    fn a_pid_match_wins_over_a_name_match() {
        // Our own pid must beat a stale/other chromatui block.
        let mixed = "Source Output #1\n\tSource: 5\n\tProperties:\n\
                     \t\tapplication.name = \"ALSA plug-in [chromatui]\"\n\
                     Source Output #2\n\tSource: 6\n\tProperties:\n\
                     \t\tapplication.name = \"weird-wrapper\"\n\
                     \t\tapplication.process.id = \"777\"\n";
        assert_eq!(parse_our_stream(mixed, "777"), Some((2, 6)));
    }

    #[test]
    fn error_distinguishes_empty_listing_from_no_match() {
        let empty = not_registered("", "1234").to_string();
        assert!(empty.contains("no capture streams"), "{empty}");

        let unmatched = not_registered(LISTING, "1234").to_string();
        assert!(unmatched.contains("1234"), "{unmatched}");
        assert!(unmatched.contains('2'), "should say how many were listed");
        assert!(!unmatched.contains("no capture streams"), "{unmatched}");
    }

    #[test]
    fn maps_source_index_to_name() {
        assert!(source_name_for_index(SOURCES, 51)
            .is_some_and(|n| n.ends_with(".monitor")));
        assert!(source_name_for_index(SOURCES, 53)
            .is_some_and(|n| !n.ends_with(".monitor")));
        assert_eq!(source_name_for_index(SOURCES, 42), None);
    }
}
