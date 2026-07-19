//! Port of the DSP + color math from spectrogram/composables/useSpectrogram.js.
//! History stores *raw* band amplitudes; sigmoid contrast and coloring are
//! applied at render time so knob changes recolor the whole visible picture,
//! exactly like the original fragment shader.

use std::sync::Arc;

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

pub fn midi_to_freq(midi: f32) -> f32 {
    440.0 * 2f32.powf((midi - 69.0) / 12.0)
}

pub fn freq_to_midi(freq: f32) -> f32 {
    69.0 + 12.0 * (freq / 440.0).log2()
}

pub struct Band {
    /// Fractional MIDI note at the band center.
    pub note: f32,
    pub freq: f32,
    bin_lo: usize,
    bin_hi: usize,
    /// Pink-noise correction (+2dB/octave vs A4), pre-scaled like the shader.
    pub pink: f32,
    /// Hue in turns [0,1): 0 = A = red, one full rainbow per octave.
    pub hue: f32,
    /// Hue quantized to a ColorMap table column.
    pub hue_idx: u16,
}

pub struct Analyzer {
    sample_rate: f32,
    fft_size: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    window_sum: f32,
    input: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    /// Exponentially smoothed linear magnitudes (Web-Audio smoothingTimeConstant).
    smoothed: Vec<f32>,
    /// Per-bin dB of `smoothed`, computed once per frame and reused by band
    /// averaging and pitch detection (was recomputed per band-bin before).
    db: Vec<f32>,
    /// Loudest bin's dB this frame (for the detection noise floor).
    peak_db: f32,
    pub bands: Vec<Band>,
}

impl Analyzer {
    pub fn new(sample_rate: f32) -> Self {
        let mut a = Analyzer {
            sample_rate,
            fft_size: 0,
            fft: RealFftPlanner::<f32>::new().plan_fft_forward(2),
            window: vec![],
            window_sum: 1.0,
            input: vec![],
            spectrum: vec![],
            smoothed: vec![],
            db: vec![],
            peak_db: -200.0,
            bands: vec![],
        };
        a.configure(8192, 128, 36, 84);
        a
    }

    /// Rebuild FFT plan and per-pixel bands. `pixels` is the number of pixels
    /// along the frequency axis; each pixel owns an equal slice of the
    /// [range_lo, range_hi] semitone span.
    pub fn configure(&mut self, fft_size: usize, pixels: usize, range_lo: i32, range_hi: i32) {
        if fft_size != self.fft_size {
            self.fft_size = fft_size;
            self.fft = RealFftPlanner::<f32>::new().plan_fft_forward(fft_size);
            self.window = (0..fft_size)
                .map(|i| {
                    let x = i as f32 / fft_size as f32;
                    0.5 - 0.5 * (2.0 * std::f32::consts::PI * x).cos()
                })
                .collect();
            self.window_sum = self.window.iter().sum();
            self.input = vec![0.0; fft_size];
            self.spectrum = vec![Complex::default(); fft_size / 2 + 1];
            self.smoothed = vec![0.0; fft_size / 2 + 1];
            self.db = vec![-200.0; fft_size / 2 + 1];
        }

        let p = pixels.max(1);
        let span = (range_hi - range_lo) as f32;
        let max_bin = fft_size / 2;
        self.bands = (0..p)
            .map(|i| {
                let note = range_lo as f32 + (i as f32 + 0.5) / p as f32 * span;
                let note_lo = range_lo as f32 + i as f32 / p as f32 * span;
                let note_hi = range_lo as f32 + (i as f32 + 1.0) / p as f32 * span;
                let freq = midi_to_freq(note);
                let f_lo = midi_to_freq(note_lo);
                let f_hi = midi_to_freq(note_hi);
                let bin_lo = ((f_lo * fft_size as f32 / self.sample_rate).floor() as usize)
                    .min(max_bin);
                let bin_hi = ((f_hi * fft_size as f32 / self.sample_rate).ceil() as usize)
                    .clamp(bin_lo, max_bin);
                let hue = ((note - 21.0) / 12.0).rem_euclid(1.0);
                Band {
                    note,
                    freq,
                    bin_lo,
                    bin_hi,
                    pink: 2.0 * (freq / 440.0).log2() * 0.01,
                    hue,
                    hue_idx: ((hue * HUE_STEPS as f32) as usize).min(HUE_STEPS - 1) as u16,
                }
            })
            .collect();
    }

    pub fn num_bands(&self) -> usize {
        self.bands.len()
    }

    /// One analysis pass: window → FFT → smoothed dB → per-band raw value,
    /// using the exact value mapping from useSpectrogram.js processFFT().
    pub fn process(&mut self, samples: &[f32], smooth: f32, out: &mut Vec<f32>) {
        debug_assert_eq!(samples.len(), self.fft_size);
        for (dst, (s, w)) in self
            .input
            .iter_mut()
            .zip(samples.iter().zip(self.window.iter()))
        {
            *dst = s * w;
        }
        let _ = self.fft.process(&mut self.input, &mut self.spectrum);

        // Normalize so a full-scale sine reads ~0 dB, like getFloatFrequencyData.
        let norm = 2.0 / self.window_sum;
        self.peak_db = -200.0;
        for ((sm, db), c) in self
            .smoothed
            .iter_mut()
            .zip(self.db.iter_mut())
            .zip(self.spectrum.iter())
        {
            let mag = c.norm() * norm;
            *sm = smooth * *sm + (1.0 - smooth) * mag;
            *db = 20.0 * sm.max(1e-10).log10();
            self.peak_db = self.peak_db.max(*db);
        }

        out.clear();
        for band in &self.bands {
            let mut sum = 0.0;
            let mut count = 0u32;
            for k in band.bin_lo..=band.bin_hi {
                sum += self.db[k];
                count += 1;
            }
            let avg_db = if count > 0 { sum / count as f32 } else { -100.0 };
            out.push(10f32.powf((avg_db + 100.0) / 100.0 - 1.0).max(0.0));
        }
    }
}

/// Sigmoid contrast from the shader: 1/(1+e^(-steep*(x-midpoint))).
pub fn sigmoid(x: f32, midpoint: f32, steep: f32) -> f32 {
    1.0 / (1.0 + (-steep * (x - midpoint)).exp())
}

/// Full per-pixel color transform from the fragment shader (exact reference;
/// the hot path uses ColorMap, which tabulates this — parity is enforced by test).
#[allow(dead_code)]
pub fn shade(raw: f32, band: &Band, midpoint: f32, steep: f32) -> (u8, u8, u8) {
    let corrected = raw + band.pink;
    let v = sigmoid(corrected, midpoint, steep);
    if v < 0.01 {
        return (0, 0, 0);
    }
    hsl_to_rgb(band.hue, v.min(1.0), v * 0.75)
}

pub const HUE_STEPS: usize = 512;
const VAL_STEPS: usize = 256;
const SIG_STEPS: usize = 2048;
/// Table domain for corrected values: pink correction makes them slightly
/// negative below A4; above SIG_MAX the sigmoid is saturated anyway.
const SIG_MIN: f32 = -0.1;
const SIG_MAX: f32 = 1.5;

/// Tabulated version of `shade` for the render hot path (~60k pixels/frame):
/// two array lookups instead of an exp() and an HSL conversion per pixel.
/// Quantization error stays below ~2/255 per channel — invisible at the
/// terminal's own 8-bit color depth.
pub struct ColorMap {
    midpoint: f32,
    steep: f32,
    /// corrected value → brightness byte (0 = the shader's black cutoff).
    sigmoid_lut: Vec<u8>,
    /// [brightness × HUE_STEPS + hue] → final RGB.
    rgb: Vec<(u8, u8, u8)>,
}

impl ColorMap {
    pub fn new() -> Self {
        let mut rgb = Vec::with_capacity(VAL_STEPS * HUE_STEPS);
        for vi in 0..VAL_STEPS {
            let v = vi as f32 / (VAL_STEPS - 1) as f32;
            for hi in 0..HUE_STEPS {
                // bucket center, halving the worst-case hue error
                let h = (hi as f32 + 0.5) / HUE_STEPS as f32;
                rgb.push(hsl_to_rgb(h, v.min(1.0), v * 0.75));
            }
        }
        let mut map = ColorMap {
            midpoint: f32::NAN,
            steep: f32::NAN,
            sigmoid_lut: vec![0; SIG_STEPS],
            rgb,
        };
        map.ensure(0.3, 20.0);
        map
    }

    /// Rebuild the (cheap) sigmoid table when the contrast knobs change.
    pub fn ensure(&mut self, midpoint: f32, steep: f32) {
        if self.midpoint == midpoint && self.steep == steep {
            return;
        }
        self.midpoint = midpoint;
        self.steep = steep;
        for (i, out) in self.sigmoid_lut.iter_mut().enumerate() {
            let x = SIG_MIN + i as f32 / (SIG_STEPS - 1) as f32 * (SIG_MAX - SIG_MIN);
            let v = sigmoid(x, midpoint, steep);
            *out = if v < 0.01 {
                0
            } else {
                (v * (VAL_STEPS - 1) as f32).round() as u8
            };
        }
    }

    #[inline]
    pub fn shade(&self, raw: f32, band: &Band) -> (u8, u8, u8) {
        let corrected = (raw + band.pink).clamp(SIG_MIN, SIG_MAX);
        let si =
            ((corrected - SIG_MIN) * ((SIG_STEPS - 1) as f32 / (SIG_MAX - SIG_MIN)) + 0.5) as usize;
        let v = self.sigmoid_lut[si] as usize;
        if v == 0 {
            return (0, 0, 0);
        }
        self.rgb[v * HUE_STEPS + band.hue_idx as usize]
    }
}

/// Bright reference color for a note (used by the ruler and readout).
pub fn note_color(midi: f32) -> (u8, u8, u8) {
    hsl_to_rgb(((midi - 21.0) / 12.0).rem_euclid(1.0), 0.9, 0.6)
}

/// HSL→RGB with the same compact formula as the shader (h in turns).
pub fn hsl_to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    let chan = |offset: f32| {
        let x = (((h * 6.0 + offset).rem_euclid(6.0) - 3.0).abs() - 1.0).clamp(0.0, 1.0);
        let v = l + s * (x - 0.5) * (1.0 - (2.0 * l - 1.0).abs());
        (v.clamp(0.0, 1.0) * 255.0).round() as u8
    };
    (chan(0.0), chan(4.0), chan(2.0))
}

/// Per-harmonic weights for the Harmonic Product Spectrum. Decreasing weights
/// make the true fundamental beat its own subharmonics (k/2, k/3…), which
/// would otherwise tie with it on pure-sine input.
const HPS_WEIGHTS: [f32; 5] = [1.0, 0.9, 0.8, 0.7, 0.6];

impl Analyzer {
    /// Fundamental-pitch estimate via a weighted Harmonic Product Spectrum.
    ///
    /// The loudest spectral peak of a real tone is often a *harmonic* (2×, 3×,
    /// 4× the pitch), which is why naive peak-picking reads an octave or an
    /// octave+fifth too high. Instead, each candidate bin k is scored by the
    /// combined (log) energy at k, 2k, 3k, 4k, 5k — only the true fundamental
    /// has energy at all of its harmonics, so it wins even when individually
    /// quieter, and even when entirely absent (the "missing fundamental").
    ///
    /// Returns (fractional midi note, frequency, sigmoid strength).
    pub fn detect_fundamental(
        &self,
        range_lo: i32,
        range_hi: i32,
        midpoint: f32,
        steep: f32,
    ) -> Option<(f32, f32, f32)> {
        let len = self.smoothed.len();
        let bin_of = |f: f32| f * self.fft_size as f32 / self.sample_rate;
        let k_lo = (bin_of(midi_to_freq(range_lo as f32)).floor() as usize).max(1);
        let k_hi = (bin_of(midi_to_freq(range_hi as f32)).ceil() as usize).min(len.saturating_sub(2));
        if k_hi <= k_lo {
            return None;
        }

        // Floor magnitudes at -80dB relative to the frame's loudest bin:
        // spectral-leakage skirts below that would otherwise let subharmonic
        // combs (k/2, k/3) outscore the true fundamental on near-sine input.
        // Works on the per-frame dB array (dB is ln times a positive constant,
        // so scores order identically and the parabola ratio is unchanged).
        let floor_db = self.peak_db - 80.0;

        let score = |k: usize| -> f32 {
            let mut sum = 0.0;
            let mut wsum = 0.0;
            for (h, &w) in HPS_WEIGHTS.iter().enumerate() {
                let i = (h + 1) * k;
                if i >= len {
                    break;
                }
                sum += w * self.db[i].max(floor_db);
                wsum += w;
            }
            sum / wsum
        };

        let (mut best_k, mut best_s) = (0, f32::MIN);
        for k in k_lo..=k_hi {
            let s = score(k);
            if s > best_s {
                best_s = s;
                best_k = k;
            }
        }

        // Sub-bin precision: parabola through the scores around the peak.
        let (yl, yc, yr) = (score(best_k - 1), best_s, score(best_k + 1));
        let denom = yl - 2.0 * yc + yr;
        let delta = if denom.abs() > 1e-9 {
            (0.5 * (yl - yr) / denom).clamp(-0.5, 0.5)
        } else {
            0.0
        };
        let freq = (best_k as f32 + delta) * self.sample_rate / self.fft_size as f32;

        // Strength from the loudest harmonic, using the same value mapping as
        // processFFT, so the visibility threshold matches the waterfall.
        let mut db = -200.0f32;
        for h in 1..=HPS_WEIGHTS.len() {
            let i = h * best_k;
            if i >= len {
                break;
            }
            db = db.max(self.db[i]);
        }
        let val = 10f32.powf((db + 100.0) / 100.0 - 1.0);
        let corrected = val + 2.0 * (freq / 440.0).log2() * 0.01;
        let v = sigmoid(corrected, midpoint, steep);
        if v < 0.15 {
            return None;
        }
        Some((freq_to_midi(freq), freq, v))
    }
}

/// Frames a challenger note must dominate before the readout switches (~150ms).
const STABLE_FRAMES: u32 = 9;
/// Frames the last note stays shown after detection drops out (~750ms).
const HOLD_FRAMES: u32 = 45;
/// EMA factor for the cents display while the note stays the same.
const CENTS_SMOOTH: f32 = 0.25;

/// Stabilizes the live note readout. Raw per-frame detection hops between
/// harmonics and neighboring bins many times a second; this keeps the shown
/// note steady: it switches only after a new note persists, smooths the cents
/// wobble, and holds through brief dropouts instead of flashing empty.
pub struct PitchTracker {
    display: Option<(f32, f32, f32)>,
    shown_note: i32,
    candidate: i32,
    candidate_frames: u32,
    hold: u32,
}

impl PitchTracker {
    pub fn new() -> Self {
        PitchTracker {
            display: None,
            shown_note: i32::MIN,
            candidate: i32::MIN,
            candidate_frames: 0,
            hold: 0,
        }
    }

    /// Feed one frame's raw detection; returns what the readout should show.
    pub fn update(&mut self, raw: Option<(f32, f32, f32)>) -> Option<(f32, f32, f32)> {
        match raw {
            Some((note, freq, v)) => {
                let rounded = note.round() as i32;
                if self.display.is_none() {
                    // nothing shown: display the first detection immediately
                    self.shown_note = rounded;
                    self.display = Some((note, freq, v));
                    self.hold = HOLD_FRAMES;
                    self.candidate_frames = 0;
                } else if rounded == self.shown_note {
                    // same note: smooth the fractional pitch, refresh the hold
                    let (dn, _, _) = self.display.unwrap();
                    let sm = dn + CENTS_SMOOTH * (note - dn);
                    self.display = Some((sm, midi_to_freq(sm), v));
                    self.hold = HOLD_FRAMES;
                    self.candidate_frames = 0;
                } else {
                    // challenger: count consecutive frames before switching
                    if rounded == self.candidate {
                        self.candidate_frames += 1;
                    } else {
                        self.candidate = rounded;
                        self.candidate_frames = 1;
                    }
                    if self.candidate_frames >= STABLE_FRAMES {
                        self.shown_note = rounded;
                        self.display = Some((note, freq, v));
                        self.hold = HOLD_FRAMES;
                        self.candidate_frames = 0;
                    } else {
                        self.tick_hold();
                    }
                }
            }
            None => {
                self.candidate_frames = 0;
                self.tick_hold();
            }
        }
        self.display
    }

    fn tick_hold(&mut self) {
        if self.hold > 0 {
            self.hold -= 1;
            if self.hold == 0 {
                self.display = None;
                self.shown_note = i32::MIN;
            }
        }
    }
}

/// Midpoint anchor for internal audio.
///
/// A mic needs *measurement* — every room, mic, and gain stage is different.
/// A digital capture does not: 0 dBFS is the same on every machine, so one
/// constant maps levels identically every time and the same track always
/// paints the same picture. (Measuring the playing content instead would move
/// the threshold with whatever happened to be playing during the pass,
/// robbing the user of a stable visual anchor.)
///
/// With the default steepness of 20: band averages around -25 dBFS (typical
/// music content, raw ~0.56) read bright, quiet beds near -50 dBFS (raw
/// ~0.32) stay dim, and digital silence (raw 0.01) renders black.
pub const INTERNAL_BASELINE: f32 = 0.40;

/// Mic auto-calibration: watch the room for ~1.25s of frames, then place the
/// midpoint just above the loudest background level observed. Internal audio
/// never runs this — it snaps to [`INTERNAL_BASELINE`], because a monitor of
/// an idle sink is exact digital silence and has no noise floor to find.
pub struct Calibration {
    frames_left: u32,
    peak: f32,
}

impl Calibration {
    pub fn new() -> Self {
        Calibration {
            frames_left: 75,
            peak: 0.0,
        }
    }

    /// Feed one raw band row; returns the new midpoint once finished.
    pub fn feed(&mut self, row: &[f32], bands: &[Band]) -> Option<f32> {
        for (v, b) in row.iter().zip(bands) {
            self.peak = self.peak.max(v + b.pink);
        }
        self.frames_left = self.frames_left.saturating_sub(1);
        (self.frames_left == 0).then(|| (self.peak + 0.06).clamp(0.05, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run a full mic calibration pass over a repeating buffer.
    fn calibrate(a: &mut Analyzer, samples: &[f32]) -> f32 {
        let mut cal = Calibration::new();
        let mut row = Vec::new();
        for _ in 0..80 {
            a.process(samples, 0.0, &mut row);
            if let Some(mid) = cal.feed(&row, &a.bands) {
                return mid;
            }
        }
        panic!("calibration never finished");
    }

    /// Music-like content: a few strong low partials over a quiet noise bed.
    fn music(n: usize, rate: f32) -> Vec<f32> {
        let mut seed = 12345u32;
        (0..n)
            .map(|i| {
                let t = i as f32 / rate;
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let noise = ((seed >> 8) as f32 / 8388608.0 - 1.0) * 0.0006;
                let tau = std::f32::consts::TAU;
                0.20 * (tau * 110.0 * t).sin()
                    + 0.10 * (tau * 220.0 * t).sin()
                    + 0.05 * (tau * 440.0 * t).sin()
                    + noise
            })
            .collect()
    }

    fn room_noise(n: usize) -> Vec<f32> {
        let mut seed = 1u32;
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                ((seed >> 8) as f32 / 8388608.0 - 1.0) * 0.0006
            })
            .collect()
    }

    /// Mic calibration is unchanged by the internal-audio work.
    #[test]
    fn mic_calibration_thresholds_just_above_room_noise() {
        let mut a = Analyzer::new(48000.0);
        let noise = room_noise(a.fft_size);
        let mid = calibrate(&mut a, &noise);
        // Just above the noise it measured, not at the clamp floor.
        assert!(
            (0.15..0.25).contains(&mid),
            "midpoint {mid} drifted from the mic baseline"
        );
        let mut row = Vec::new();
        a.process(&noise, 0.0, &mut row);
        let loudest = row
            .iter()
            .zip(&a.bands)
            .fold(f32::MIN, |m, (v, b)| m.max(v + b.pink));
        assert!(mid > loudest, "midpoint {mid} must sit above noise {loudest}");
    }

    /// The reported bug: an idle sink's monitor is exact digital silence, and
    /// calibrating against it put the midpoint below every real signal,
    /// saturating the whole screen. The fixed baseline must render silence
    /// as black instead.
    #[test]
    fn internal_baseline_blanks_digital_silence() {
        let mut a = Analyzer::new(48000.0);
        let silence = vec![0.0; a.fft_size];
        let mut row = Vec::new();
        a.process(&silence, 0.0, &mut row);
        let max = row
            .iter()
            .zip(&a.bands)
            .fold(f32::MIN, |m, (v, b)| {
                m.max(sigmoid(v + b.pink, INTERNAL_BASELINE, 20.0))
            });
        assert!(max < 0.01, "digital silence renders at brightness {max}");
    }

    /// The baseline is a *fixed* anchor, so it has to yield real contrast on
    /// typical music content without any measurement: played notes bright,
    /// empty spectrum dark.
    #[test]
    fn internal_baseline_leaves_contrast_on_music() {
        let mut a = Analyzer::new(48000.0);
        let samples = music(a.fft_size, 48000.0);
        let mid = INTERNAL_BASELINE;

        let mut row = Vec::new();
        a.process(&samples, 0.0, &mut row);
        let brightness: Vec<f32> = row
            .iter()
            .zip(&a.bands)
            .map(|(v, b)| sigmoid(v + b.pink, mid, 20.0))
            .collect();

        // The 110 Hz partial (A2, MIDI 45) must read as lit up.
        let loud = a
            .bands
            .iter()
            .position(|b| (b.note - 45.0).abs() < 0.5)
            .expect("A2 is inside the default range");
        assert!(
            brightness[loud] > 0.8,
            "played note only reached {}",
            brightness[loud]
        );

        // And most of the spectrum, which holds nothing, must stay dark.
        let dark = brightness.iter().filter(|&&v| v < 0.2).count();
        assert!(
            dark > brightness.len() / 2,
            "only {dark}/{} bands stayed dark — screen is saturated",
            brightness.len()
        );
    }

    #[test]
    fn pitch_math() {
        assert!((midi_to_freq(69.0) - 440.0).abs() < 1e-3);
        assert!((midi_to_freq(60.0) - 261.626).abs() < 1e-2);
        assert!((freq_to_midi(880.0) - 81.0).abs() < 1e-4);
    }

    #[test]
    fn sigmoid_midpoint_is_half() {
        assert!((sigmoid(0.3, 0.3, 20.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(1.0, 0.3, 20.0) > 0.99);
        assert!(sigmoid(0.0, 0.3, 20.0) < 0.01);
    }

    #[test]
    fn a_is_red() {
        // Hue 0 (any A) must come out red-dominant.
        let (r, g, b) = hsl_to_rgb(0.0, 1.0, 0.5);
        assert!(r > 200 && g < 60 && b < 60, "got {r},{g},{b}");
    }

    /// Synthesize one FFT window of a tone made of (freq, amplitude) partials.
    fn tone(n: usize, sr: f32, partials: &[(f32, f32)]) -> Vec<f32> {
        (0..n)
            .map(|i| {
                partials
                    .iter()
                    .map(|(f, a)| (2.0 * std::f32::consts::PI * f * i as f32 / sr).sin() * a)
                    .sum()
            })
            .collect()
    }

    fn analyzed(partials: &[(f32, f32)]) -> Analyzer {
        let (sr, n) = (48000.0, 8192);
        let mut an = Analyzer::new(sr);
        an.configure(n, 240, 36, 84); // C2..C6
        let samples = tone(n, sr, partials);
        let mut row = Vec::new();
        an.process(&samples, 0.0, &mut row);
        an
    }

    #[test]
    fn detects_a4_sine() {
        let an = analyzed(&[(440.0, 0.5)]);
        let (note, freq, v) = an
            .detect_fundamental(36, 84, 0.3, 20.0)
            .expect("no pitch detected");
        assert!((note - 69.0).abs() < 0.3, "note {note}");
        assert!((freq - 440.0).abs() < 8.0, "freq {freq}");
        assert!(v > 0.5);
    }

    #[test]
    fn harmonic_rich_tone_reports_fundamental_not_harmonic() {
        // A2 (110 Hz) with the 2nd and 3rd harmonics LOUDER than the
        // fundamental — typical for voice and piano. Naive loudest-band
        // detection reports A3 (220) or E4 (330); the readout must say A2.
        let an = analyzed(&[(110.0, 0.2), (220.0, 0.5), (330.0, 0.4), (440.0, 0.25)]);
        let (note, freq, _v) = an
            .detect_fundamental(36, 84, 0.3, 20.0)
            .expect("no pitch detected");
        assert!((freq - 110.0).abs() < 4.0, "freq {freq} (octave/fifth error)");
        assert!((note - 45.0).abs() < 0.5, "note {note}");
    }

    #[test]
    fn high_note_not_reported_an_octave_down() {
        // C5 with normal decaying harmonics: the subharmonic comb at C4 must
        // not win (HPS's classic octave-down failure mode).
        let an = analyzed(&[(523.25, 0.5), (1046.5, 0.3), (1569.75, 0.2)]);
        let (_, freq, _) = an
            .detect_fundamental(36, 84, 0.3, 20.0)
            .expect("no pitch detected");
        assert!((freq - 523.25).abs() < 12.0, "freq {freq}");
    }

    #[test]
    fn missing_fundamental_still_resolved() {
        // Even with NO energy at 110 Hz at all, the shared spacing of the
        // harmonics implies the fundamental (how human hearing works too).
        let an = analyzed(&[(220.0, 0.4), (330.0, 0.4), (440.0, 0.3), (550.0, 0.2)]);
        let (_, freq, _) = an
            .detect_fundamental(36, 84, 0.3, 20.0)
            .expect("no pitch detected");
        assert!((freq - 110.0).abs() < 4.0, "freq {freq}");
    }

    fn det(note: f32) -> Option<(f32, f32, f32)> {
        Some((note, midi_to_freq(note), 0.8))
    }

    #[test]
    fn tracker_shows_first_detection_immediately() {
        let mut t = PitchTracker::new();
        let shown = t.update(det(69.0)).unwrap();
        assert_eq!(shown.0.round() as i32, 69);
    }

    #[test]
    fn tracker_ignores_rapid_flicker() {
        let mut t = PitchTracker::new();
        t.update(det(69.0));
        // alternate A4 / C5 every frame for a second: display must stay A4
        for i in 0..60 {
            let shown = t.update(det(if i % 2 == 0 { 72.0 } else { 69.0 })).unwrap();
            assert_eq!(shown.0.round() as i32, 69, "flipped at frame {i}");
        }
    }

    #[test]
    fn tracker_switches_after_sustained_new_note() {
        let mut t = PitchTracker::new();
        t.update(det(69.0));
        let mut shown = t.update(det(72.0)).unwrap();
        assert_eq!(shown.0.round() as i32, 69, "switched too early");
        for _ in 0..STABLE_FRAMES {
            shown = t.update(det(72.0)).unwrap();
        }
        assert_eq!(shown.0.round() as i32, 72);
    }

    #[test]
    fn tracker_holds_through_dropouts_then_clears() {
        let mut t = PitchTracker::new();
        t.update(det(69.0));
        for _ in 0..(HOLD_FRAMES - 1) {
            assert!(t.update(None).is_some(), "cleared during hold window");
        }
        assert!(t.update(None).is_none(), "should clear after hold expires");
        // next detection shows immediately again
        assert!(t.update(det(60.0)).is_some());
    }

    #[test]
    fn colormap_matches_exact_shade() {
        let an = analyzed(&[(440.0, 0.5)]);
        let mut map = ColorMap::new();
        for &(midpoint, steep) in &[(0.3f32, 20.0f32), (0.1, 3.0), (0.8, 40.0)] {
            map.ensure(midpoint, steep);
            for band in an.bands.iter().step_by(7) {
                for i in 0..=100 {
                    let raw = i as f32 * 0.012;
                    let exact = shade(raw, band, midpoint, steep);
                    let fast = map.shade(raw, band);
                    for (a, b) in [
                        (exact.0, fast.0),
                        (exact.1, fast.1),
                        (exact.2, fast.2),
                    ] {
                        assert!(
                            (a as i16 - b as i16).abs() <= 4,
                            "raw {raw} mid {midpoint} steep {steep}: exact {exact:?} vs lut {fast:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn colormap_speed_vs_exact() {
        // Not an assertion-based test: prints the per-frame cost of shading
        // 60k lit pixels (a fullscreen 250x60 quad frame) both ways.
        // Run: cargo test --release colormap_speed -- --nocapture
        let an = analyzed(&[(440.0, 0.5)]);
        let map = ColorMap::new();
        let n = 60_000;
        let bands = &an.bands;

        let t = std::time::Instant::now();
        let mut acc = 0u32;
        for i in 0..n {
            let b = &bands[i % bands.len()];
            let c = shade(0.2 + (i % 7) as f32 * 0.1, b, 0.3, 20.0);
            acc = acc.wrapping_add(c.0 as u32);
        }
        let exact_us = t.elapsed().as_micros();

        let t = std::time::Instant::now();
        for i in 0..n {
            let b = &bands[i % bands.len()];
            let c = map.shade(0.2 + (i % 7) as f32 * 0.1, b);
            acc = acc.wrapping_add(c.0 as u32);
        }
        let lut_us = t.elapsed().as_micros();
        println!("shade 60k lit pixels: exact {exact_us}µs vs lut {lut_us}µs (acc {acc})");
    }

    #[test]
    fn silence_detects_nothing() {
        let an = analyzed(&[]);
        assert!(an.detect_fundamental(36, 84, 0.3, 20.0).is_none());
    }
}
