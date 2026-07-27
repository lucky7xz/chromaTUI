mod analysis;
mod audio;
mod config;
mod controls;
mod render;

use std::collections::VecDeque;
use std::io::{self, BufWriter, Stdout, Write};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use ratatui::style::Color;

use analysis::{Analyzer, Calibration, ColorMap, PitchTracker, INTERNAL_BASELINE};
use audio::InputSource;
use controls::{Params, PARAMS};

/// Below this the app shows the "go fullscreen" screen instead of the waterfall.
/// Below this the waterfall still renders, it just gets coarse — the floor is
/// the classic 80×24 so any terminal works, but bigger is much better.
pub const MIN_COLS: u16 = 80;
pub const MIN_ROWS: u16 = 24;

const FRAME: Duration = Duration::from_micros(16_667); // 60 fps
const HISTORY_CAP: usize = 1024;
/// How long a transient status message stays on screen.
const TOAST_TTL: Duration = Duration::from_secs(2);

pub struct App {
    pub params: Params,
    pub focus: usize,
    pub paused: bool,
    pub analyzer: Analyzer,
    pub colors: ColorMap,
    /// Raw band rows, newest at the front.
    pub history: VecDeque<Vec<f32>>,
    pub calibration: Option<Calibration>,
    pub current_pitch: Option<(f32, f32, f32)>,
    pitch_tracker: PitchTracker,
    pub help_visible: bool,
    pub wheel_visible: bool,
    pub pending_reset: bool,
    /// Transient message shown in the status line; expires on its own.
    pub toast: Option<(String, Color, Instant)>,
    audio: audio::AudioInput,
    samples: Vec<f32>,
    row: Vec<f32>,
    scroll_acc: f32,
    size: (u16, u16),
    needs_rebuild: bool,
    quit: bool,
}

impl App {
    fn new(params: Params, audio: audio::AudioInput) -> Self {
        let analyzer = Analyzer::new(audio.sample_rate);
        App {
            samples: vec![0.0; params.fft_size()],
            params,
            focus: 0,
            paused: false,
            analyzer,
            colors: ColorMap::new(),
            history: VecDeque::new(),
            calibration: None,
            current_pitch: None,
            pitch_tracker: PitchTracker::new(),
            help_visible: false,
            wheel_visible: false,
            pending_reset: false,
            // The server may have restored last session's route; make a
            // non-default start impossible to miss.
            toast: (audio.source == audio::InputSource::Internal).then(|| {
                (
                    "input: internal audio (restored from last session)".to_string(),
                    Color::Cyan,
                    Instant::now(),
                )
            }),
            audio,
            row: Vec::new(),
            scroll_acc: 0.0,
            size: (0, 0),
            needs_rebuild: true,
            quit: false,
        }
    }

    pub fn device_name(&self) -> &str {
        &self.audio.device_name
    }

    pub fn source_label(&self) -> &'static str {
        self.audio.source.label()
    }

    pub fn source(&self) -> InputSource {
        self.audio.source
    }

    fn set_toast(&mut self, text: String, color: Color) {
        self.toast = Some((text, color, Instant::now()));
    }

    /// Pixels along the frequency axis for the current orientation/size.
    /// Quadrant cells give 2 pixels per cell on both axes.
    fn freq_pixels(&self) -> usize {
        if self.params.horizontal {
            self.size.1 as usize * 2
        } else {
            self.size.0 as usize * 2
        }
    }

    fn rebuild(&mut self) {
        self.analyzer.configure(
            self.params.fft_size(),
            self.freq_pixels(),
            self.params.range_lo,
            self.params.range_hi,
        );
        self.samples.resize(self.params.fft_size(), 0.0);
        self.history.clear();
        self.scroll_acc = 0.0;
        self.needs_rebuild = false;
    }

    fn tick(&mut self, size: (u16, u16)) {
        if size != self.size {
            self.size = size;
            self.needs_rebuild = true;
        }
        if self.size.0 < MIN_COLS || self.size.1 < MIN_ROWS {
            return;
        }
        if self.needs_rebuild {
            self.rebuild();
        }
        self.colors.ensure(self.params.midpoint, self.params.steep);

        if let Some((_, _, at)) = &self.toast {
            if at.elapsed() >= TOAST_TTL {
                self.toast = None;
            }
        }

        self.audio.copy_latest(&mut self.samples);
        self.analyzer
            .process(&self.samples, self.params.smooth, &mut self.row);

        let raw_pitch = self.analyzer.detect_fundamental(
            self.params.range_lo,
            self.params.range_hi,
            self.params.midpoint,
            self.params.steep,
        );
        self.current_pitch = self.pitch_tracker.update(raw_pitch);

        if let Some(cal) = &mut self.calibration {
            if let Some(midpoint) = cal.feed(&self.row, &self.analyzer.bands) {
                self.params.midpoint = midpoint;
                self.calibration = None;
            }
        }

        if !self.paused {
            self.scroll_acc += self.params.speed();
            while self.scroll_acc >= 1.0 {
                self.history.push_front(self.row.clone());
                self.history.truncate(HISTORY_CAP);
                self.scroll_acc -= 1.0;
            }
        }
    }

    fn on_key(&mut self, key: KeyEvent) {
        // The reset confirmation modal captures all input until answered.
        if self.pending_reset {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') => {
                    self.params = Params::default();
                    self.needs_rebuild = true;
                    self.pending_reset = false;
                }
                KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                    self.pending_reset = false
                }
                _ => {}
            }
            return;
        }
        let coarse = key.modifiers.contains(KeyModifiers::SHIFT);
        match key.code {
            KeyCode::Char('q') => self.quit = true,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.quit = true
            }
            KeyCode::Char('?') => {
                self.help_visible = !self.help_visible;
                self.wheel_visible = false;
            }
            KeyCode::Char('f') => {
                self.wheel_visible = !self.wheel_visible;
                self.help_visible = false;
            }
            KeyCode::Esc => {
                self.help_visible = false;
                self.wheel_visible = false;
            }
            KeyCode::Char('i') => match self.audio.toggle_source() {
                Ok(()) => {
                    let msg = match self.audio.source {
                        InputSource::Mic => "input: microphone (c recalibrates)".into(),
                        InputSource::Internal => {
                            "input: internal audio (c sets the baseline)".into()
                        }
                    };
                    self.set_toast(msg, Color::Cyan);
                }
                Err(e) => self.set_toast(format!("input switch failed: {e}"), Color::Red),
            },
            KeyCode::Char('r') => self.pending_reset = true,
            KeyCode::Char(' ') => self.paused = !self.paused,
            KeyCode::Enter => {
                self.history.clear();
                self.scroll_acc = 0.0;
            }
            KeyCode::Char('o') => {
                self.params.horizontal = !self.params.horizontal;
                self.needs_rebuild = true;
            }
            KeyCode::Char('c') => match self.audio.source {
                InputSource::Mic => {
                    if self.calibration.is_none() {
                        self.calibration = Some(Calibration::new());
                    }
                }
                // Digital audio has a fixed reference (0 dBFS on every
                // machine), so snap to the constant anchor instead of
                // measuring whatever happens to be playing.
                InputSource::Internal => {
                    self.params.midpoint = INTERNAL_BASELINE;
                    self.set_toast("midpoint set to internal baseline".into(), Color::Cyan);
                }
            },
            KeyCode::Tab => self.focus = (self.focus + 1) % PARAMS.len(),
            KeyCode::BackTab => self.focus = (self.focus + PARAMS.len() - 1) % PARAMS.len(),
            KeyCode::Char(d @ '1'..='7') => self.focus = d as usize - '1' as usize,
            KeyCode::Up => self.focus = self.focus.saturating_sub(1),
            KeyCode::Down => self.focus = (self.focus + 1).min(PARAMS.len() - 1),
            KeyCode::Right => {
                if self.params.adjust(PARAMS[self.focus], 1, coarse) {
                    self.needs_rebuild = true;
                }
            }
            KeyCode::Left => {
                if self.params.adjust(PARAMS[self.focus], -1, coarse) {
                    self.needs_rebuild = true;
                }
            }
            _ => {}
        }
    }
}

type Term = Terminal<CrosstermBackend<BufWriter<Stdout>>>;

/// Like ratatui::init(), but with a large write buffer so a frame's ANSI
/// stream goes out in a few big writes instead of many small ones.
fn init_terminal() -> Result<Term> {
    enable_raw_mode()?;
    let mut out = BufWriter::with_capacity(1 << 18, io::stdout());
    crossterm::execute!(out, EnterAlternateScreen)?;
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        restore_terminal();
        hook(info);
    }));
    Ok(Terminal::new(CrosstermBackend::new(out))?)
}

fn restore_terminal() {
    let _ = disable_raw_mode();
    let mut out = io::stdout();
    let _ = crossterm::execute!(out, LeaveAlternateScreen, crossterm::cursor::Show);
    let _ = out.flush();
}

/// Per-frame timing accumulator, reported when CHROMATUI_STATS is set.
#[derive(Default)]
struct FrameStats {
    frames: u64,
    tick_us: u64,
    draw_us: u64,
    max_tick_us: u64,
    max_draw_us: u64,
    overruns: u64,
}

impl FrameStats {
    fn report(&self) {
        if self.frames == 0 {
            return;
        }
        eprintln!(
            "chromatui stats: {} frames | tick avg {}µs max {}µs | draw avg {}µs max {}µs | {} overruns (>16.6ms)",
            self.frames,
            self.tick_us / self.frames,
            self.max_tick_us,
            self.draw_us / self.frames,
            self.max_draw_us,
            self.overruns
        );
    }
}

fn main() -> Result<()> {
    let params = config::load();
    let audio = audio::AudioInput::start()?;
    let mut app = App::new(params, audio);

    let mut terminal = init_terminal()?;
    let mut stats = FrameStats::default();
    let result = run(&mut terminal, &mut app, &mut stats);
    restore_terminal();

    if std::env::var_os("CHROMATUI_STATS").is_some() {
        stats.report();
    }
    config::save(&app.params);
    result
}

fn run(terminal: &mut Term, app: &mut App, stats: &mut FrameStats) -> Result<()> {
    let mut next_frame = Instant::now();
    while !app.quit {
        while event::poll(Duration::ZERO)? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => app.on_key(key),
                _ => {}
            }
        }

        let frame_start = Instant::now();
        let size = terminal.size()?;
        app.tick((size.width, size.height));
        let ticked = Instant::now();
        terminal.draw(|f| render::draw(f, app))?;
        let drawn = Instant::now();

        stats.frames += 1;
        let tick_us = ticked.duration_since(frame_start).as_micros() as u64;
        let draw_us = drawn.duration_since(ticked).as_micros() as u64;
        stats.tick_us += tick_us;
        stats.draw_us += draw_us;
        stats.max_tick_us = stats.max_tick_us.max(tick_us);
        stats.max_draw_us = stats.max_draw_us.max(draw_us);

        next_frame += FRAME;
        let now = Instant::now();
        if next_frame > now {
            std::thread::sleep(next_frame - now);
        } else {
            stats.overruns += 1;
            next_frame = now; // fell behind; don't try to catch up
        }
    }
    Ok(())
}
