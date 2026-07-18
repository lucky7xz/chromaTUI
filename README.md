# chromaTUI

The [Chromatone spectrogram](https://spectrogram.chromatone.center) rewritten in Rust for the terminal (ratatui). Speak at the mic and watch pitch the color: every A is red, one full rainbow per octave.

**Use it fullscreen** — below 140×35 cells it shows a "too small" screen. Needs a truecolor terminal and a microphone.

## Build & run

Needs a Rust toolchain ([rustup](https://rustup.rs)) and a working microphone.

```sh
git clone https://github.com/lucky7xz/chromaTUI.git && cd chromaTUI
cargo run --release      # or: cargo build --release && ./target/release/chromatui
```

`cargo run` (debug) is fine too — the dev profile is built at `opt-level = 2` so it stays smooth. On Linux you need ALSA headers for cpal: `sudo apt install libasound2-dev` (or `alsa-lib-devel` on Fedora).

## Terminals

The whole screen is repainted every frame, so your terminal emulator matters more than one wouldd expect. In my own testing : **ghostty > wezterm > gnome-terminal**. Run with `CHROMATUI_STATS=1` to print frame-time stats on exit and compare on your own machine.

## Keys

| Key | Action |
|---|---|
| `↑`/`↓` / `Tab` / `1`–`7` | select a control |
| `←`/`→` (Shift = coarse) | adjust it |
| `c` | auto-calibrate sensitivity (stay quiet ~1s) |
| `?` | help overlay: keys + what each setting does |
| `f` | note-color wheel: which note is which color |
| `o` | flip orientation (pitch vertical ↔ horizontal) |
| `Space` / `Enter` | pause / clear |
| `r` | reset all settings to defaults (asks y/n) |
| `q` | quit (saves settings) |

## Controls

- **fft size** — pitch sharpness vs. responsiveness: bigger = tells close (low) notes apart better but reacts slower.
- **smooth** — blends frames; 0 = instant and jittery, higher = calmer but smeared.
- **speed** — waterfall scroll rate, ¼×–4×; slower shows more history, faster stretches detail.
- **midpoint** — sensitivity: how loud a sound must be to show up. `c` sets it for your room automatically.
- **steep** — contrast: high = crisp on/off lines, low = soft gradients showing loudness.
- **range lo/hi** — pitch span on screen; default C2–C6 (singing voice). Narrower = more detail per note.

Settings persist in `~/.config/chromatui/config.toml`. The original web app lives in `research/spectrogram/` and defines all the math this port reproduces.

> **AI notice:** this project was built with heavy use of AI coding assistants. The design decisions are mine; a lot of the code was written by an LLM under review.

## License

MIT — see [LICENSE](LICENSE).
