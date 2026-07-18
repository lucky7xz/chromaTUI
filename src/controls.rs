use serde::{Deserialize, Serialize};

/// Scroll speeds in waterfall rows per frame. Sub-1 values advance the
/// waterfall every 2nd/4th frame so more history fits on screen.
pub const SPEEDS: [f32; 6] = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0];
pub const SPEED_LABELS: [&str; 6] = ["¼", "½", "1", "2", "3", "4"];

pub const MIN_NOTE: i32 = 21; // A0
pub const MAX_NOTE: i32 = 132; // C9
pub const MIN_SPAN: i32 = 12; // at least one octave visible

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Param {
    Fft,
    Smooth,
    Speed,
    Midpoint,
    Steep,
    RangeLo,
    RangeHi,
}

pub const PARAMS: [Param; 7] = [
    Param::Fft,
    Param::Smooth,
    Param::Speed,
    Param::Midpoint,
    Param::Steep,
    Param::RangeLo,
    Param::RangeHi,
];

impl Param {
    pub fn label(self) -> &'static str {
        match self {
            Param::Fft => "fft size",
            Param::Smooth => "smooth",
            Param::Speed => "speed",
            Param::Midpoint => "midpoint",
            Param::Steep => "steep",
            Param::RangeLo => "range lo",
            Param::RangeHi => "range hi",
        }
    }

    /// Plain-language explanation shown in the help overlay. Written so a
    /// user can paste it into an LLM to dig deeper.
    pub fn explain(self) -> &'static str {
        match self {
            Param::Fft => {
                "how much audio each analysis looks at (the FFT window). Bigger tells \
                 close-together and low notes apart better but reacts slower; smaller \
                 is snappier but blurs low pitches."
            }
            Param::Smooth => {
                "blends each frame with what was already on screen. 0 is instant but \
                 jittery; higher is calmer but smears fast changes and leaves ghosts."
            }
            Param::Speed => {
                "how fast the waterfall scrolls. Slow keeps more seconds of history \
                 visible; fast stretches fine detail like vibrato across more screen."
            }
            Param::Midpoint => {
                "how loud a sound must be to show up (a noise gate). Raise it to hide \
                 room noise, lower it to see quiet sounds. Press c to set it \
                 automatically from your room's noise floor."
            }
            Param::Steep => {
                "contrast around the midpoint. High gives crisp on/off lines; low \
                 gives soft gradients that show how loud each note is."
            }
            Param::RangeLo => {
                "lowest note on screen. A narrower range gives every note more \
                 pixels, so pitch detail gets finer."
            }
            Param::RangeHi => {
                "highest note on screen. A narrower range gives every note more \
                 pixels, so pitch detail gets finer."
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Params {
    pub fft_exp: u32,
    pub smooth: f32,
    pub speed_idx: usize,
    pub midpoint: f32,
    pub steep: f32,
    pub range_lo: i32,
    pub range_hi: i32,
    pub horizontal: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            fft_exp: 13,
            smooth: 0.0,
            speed_idx: 2,
            midpoint: 0.3,
            steep: 20.0,
            range_lo: 36, // C2
            range_hi: 84, // C6
            horizontal: true,
        }
    }
}

impl Params {
    pub fn fft_size(&self) -> usize {
        1 << self.fft_exp
    }

    pub fn speed(&self) -> f32 {
        SPEEDS[self.speed_idx]
    }

    /// Clamp everything into a valid state (used after loading config).
    pub fn sanitize(&mut self) {
        self.fft_exp = self.fft_exp.clamp(12, 14);
        self.smooth = self.smooth.clamp(0.0, 1.0);
        self.speed_idx = self.speed_idx.min(SPEEDS.len() - 1);
        self.midpoint = self.midpoint.clamp(0.0, 1.0);
        self.steep = self.steep.clamp(3.0, 40.0);
        self.range_lo = self.range_lo.clamp(MIN_NOTE, MAX_NOTE - MIN_SPAN);
        self.range_hi = self.range_hi.clamp(self.range_lo + MIN_SPAN, MAX_NOTE);
    }

    /// Nudge one parameter. Returns true when the change requires
    /// rebuilding the analysis bands (fft size or pitch range).
    pub fn adjust(&mut self, param: Param, dir: i32, coarse: bool) -> bool {
        let d = dir as f32;
        match param {
            Param::Fft => {
                self.fft_exp = (self.fft_exp as i32 + dir).clamp(12, 14) as u32;
                true
            }
            Param::Smooth => {
                let step = if coarse { 0.1 } else { 0.01 };
                self.smooth = (self.smooth + d * step).clamp(0.0, 1.0);
                false
            }
            Param::Speed => {
                self.speed_idx =
                    (self.speed_idx as i32 + dir).clamp(0, SPEEDS.len() as i32 - 1) as usize;
                false
            }
            Param::Midpoint => {
                let step = if coarse { 0.05 } else { 0.01 };
                self.midpoint = (self.midpoint + d * step).clamp(0.0, 1.0);
                false
            }
            Param::Steep => {
                let step = if coarse { 5.0 } else { 0.5 };
                self.steep = (self.steep + d * step).clamp(3.0, 40.0);
                false
            }
            Param::RangeLo => {
                let step = if coarse { 12 } else { 1 };
                self.range_lo =
                    (self.range_lo + dir * step).clamp(MIN_NOTE, self.range_hi - MIN_SPAN);
                true
            }
            Param::RangeHi => {
                let step = if coarse { 12 } else { 1 };
                self.range_hi =
                    (self.range_hi + dir * step).clamp(self.range_lo + MIN_SPAN, MAX_NOTE);
                true
            }
        }
    }

    pub fn value_str(&self, param: Param) -> String {
        match param {
            Param::Fft => format!("{}", self.fft_size()),
            Param::Smooth => format!("{:.2}", self.smooth),
            Param::Speed => SPEED_LABELS[self.speed_idx].to_string(),
            Param::Midpoint => format!("{:.2}", self.midpoint),
            Param::Steep => format!("{:.1}", self.steep),
            Param::RangeLo => note_name(self.range_lo),
            Param::RangeHi => note_name(self.range_hi),
        }
    }
}

const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

pub fn note_name(midi: i32) -> String {
    let idx = midi.rem_euclid(12) as usize;
    let octave = midi.div_euclid(12) - 1;
    format!("{}{}", NOTE_NAMES[idx], octave)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_names() {
        assert_eq!(note_name(60), "C4");
        assert_eq!(note_name(69), "A4");
        assert_eq!(note_name(21), "A0");
        assert_eq!(note_name(120), "C9");
        // The JS source comments call MIDI 132 "C9", but 132 is C10 (~16.7 kHz).
        assert_eq!(note_name(132), "C10");
    }

    #[test]
    fn adjust_clamps_and_flags_rebuild() {
        let mut p = Params::default();
        assert!(p.adjust(Param::Fft, 5, false));
        assert_eq!(p.fft_exp, 14);
        assert!(!p.adjust(Param::Midpoint, -100, false));
        assert_eq!(p.midpoint, 0.0);
        // range lo can never cross range hi minus an octave
        for _ in 0..200 {
            p.adjust(Param::RangeLo, 1, true);
        }
        assert!(p.range_lo <= p.range_hi - MIN_SPAN);
    }

    #[test]
    fn sanitize_repairs_bad_config() {
        let mut p = Params {
            fft_exp: 99,
            smooth: 7.0,
            speed_idx: 42,
            midpoint: -3.0,
            steep: 1000.0,
            range_lo: 130,
            range_hi: 20,
            horizontal: true,
        };
        p.sanitize();
        assert!(p.fft_exp <= 14 && p.range_lo + MIN_SPAN <= p.range_hi);
        assert!(p.speed_idx < SPEEDS.len());
    }

    #[test]
    fn every_param_has_an_explanation() {
        for p in PARAMS {
            assert!(p.explain().len() > 40, "{} explanation too short", p.label());
        }
    }
}
