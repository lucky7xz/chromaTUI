use std::fs;
use std::path::PathBuf;

use crate::controls::Params;

pub fn path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("chromatui").join("config.toml"))
}

/// Load saved params, falling back to defaults; always sanitized.
pub fn load() -> Params {
    let mut params: Params = path()
        .and_then(|p| fs::read_to_string(p).ok())
        .and_then(|s| toml::from_str(&s).ok())
        .unwrap_or_default();
    params.sanitize();
    params
}

pub fn save(params: &Params) {
    let Some(p) = path() else { return };
    if let Some(dir) = p.parent() {
        let _ = fs::create_dir_all(dir);
    }
    if let Ok(s) = toml::to_string_pretty(params) {
        let _ = fs::write(p, s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::Param;

    #[test]
    fn toml_round_trip() {
        let mut p = Params::default();
        p.adjust(Param::Midpoint, 5, true);
        p.adjust(Param::Fft, 1, false);
        let s = toml::to_string_pretty(&p).unwrap();
        let q: Params = toml::from_str(&s).unwrap();
        assert_eq!(q.fft_exp, p.fft_exp);
        assert!((q.midpoint - p.midpoint).abs() < 1e-6);
    }

    #[test]
    fn partial_toml_uses_defaults() {
        let q: Params = toml::from_str("midpoint = 0.5\n").unwrap();
        assert!((q.midpoint - 0.5).abs() < 1e-6);
        assert_eq!(q.fft_exp, Params::default().fft_exp);
    }
}
