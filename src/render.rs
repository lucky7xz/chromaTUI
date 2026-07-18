//! All drawing. The waterfall uses quadrant-block cells: each cell shows a
//! 2×2 pixel patch via the 16 Block Elements glyphs. A cell only has two
//! colors (fg/bg), so the four pixels are split into bright/dark clusters at
//! their mean luminance; the glyph marks which quadrants belong to the bright
//! cluster. Neighboring spectrogram pixels are similar, so the fit is close.

use ratatui::buffer::Buffer;
use ratatui::layout::{Alignment, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};
use ratatui::Frame;

use crate::analysis::note_color;
use crate::controls::{note_name, PARAMS};
use crate::{App, MIN_COLS, MIN_ROWS};

pub fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    if area.width < MIN_COLS || area.height < MIN_ROWS {
        draw_too_small(f, area);
        return;
    }
    draw_waterfall(f.buffer_mut(), area, app);
    draw_ruler(f.buffer_mut(), area, app);
    draw_readout(f, area, app);
    draw_status(f, area, app);
    draw_panel(f, area, app);
    if app.help_visible {
        draw_help(f, area, app);
    }
    if app.pending_reset {
        draw_reset_prompt(f, area);
    }
}

fn draw_too_small(f: &mut Frame, area: Rect) {
    let lines = vec![
        Line::from(Span::styled(
            "chromaTUI",
            Style::default().add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(format!(
            "Terminal is {}×{} — too small.",
            area.width, area.height
        )),
        Line::from(format!(
            "Make the terminal fullscreen (needs at least {}×{}).",
            MIN_COLS, MIN_ROWS
        )),
    ];
    let y = area.height.saturating_sub(4) / 2;
    let rect = Rect::new(area.x, area.y + y, area.width, 4.min(area.height));
    f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), rect);
}

type Rgb = (u8, u8, u8);

/// Glyph for each 4-bit mask of "bright" quadrants: tl=8, tr=4, bl=2, br=1.
const QUADS: [char; 16] = [
    ' ', '▗', '▖', '▄', '▝', '▐', '▞', '▟', '▘', '▚', '▌', '▙', '▀', '▜', '▛', '█',
];

fn luma(c: Rgb) -> f32 {
    0.299 * c.0 as f32 + 0.587 * c.1 as f32 + 0.114 * c.2 as f32
}

/// Fit 4 pixels [tl, tr, bl, br] into one cell: split bright/dark at the mean
/// luminance, average each cluster's color, pick the matching quadrant glyph.
fn quad_cell(px: [Rgb; 4]) -> (char, Rgb, Rgb) {
    let l = px.map(luma);
    let mean = l.iter().sum::<f32>() / 4.0;
    let mut mask = 0usize;
    let mut hi = [0u32; 3];
    let mut lo = [0u32; 3];
    let (mut hi_n, mut lo_n) = (0u32, 0u32);
    for i in 0..4 {
        let (r, g, b) = px[i];
        if l[i] >= mean {
            mask |= 8 >> i;
            hi[0] += r as u32;
            hi[1] += g as u32;
            hi[2] += b as u32;
            hi_n += 1;
        } else {
            lo[0] += r as u32;
            lo[1] += g as u32;
            lo[2] += b as u32;
            lo_n += 1;
        }
    }
    let avg = |c: [u32; 3], n: u32| -> Rgb {
        ((c[0] / n) as u8, (c[1] / n) as u8, (c[2] / n) as u8)
    };
    let fg = avg(hi, hi_n.max(1)); // hi always has the max-luma pixel
    let bg = if lo_n > 0 { avg(lo, lo_n) } else { fg };
    (QUADS[mask], fg, bg)
}

fn draw_waterfall(buf: &mut Buffer, area: Rect, app: &App) {
    let bands = &app.analyzer.bands;
    let w = area.width as usize;
    let h = area.height as usize;

    // One spectrogram pixel: history row `t` (0 = newest), band index → color.
    let pix = |t: usize, band_idx: usize| -> Rgb {
        app.history
            .get(t)
            .and_then(|row| {
                Some(app.colors.shade(*row.get(band_idx)?, bands.get(band_idx)?))
            })
            .unwrap_or((0, 0, 0))
    };

    for y in 0..h {
        for x in 0..w {
            let mut px = [(0, 0, 0); 4];
            for dy in 0..2 {
                for dx in 0..2 {
                    px[dy * 2 + dx] = if app.params.horizontal {
                        // 2 time steps per column (newest right), pitch vertical
                        let t = 2 * w - 1 - (2 * x + dx);
                        let band = 2 * h - 1 - (2 * y + dy);
                        pix(t, band)
                    } else {
                        // 2 bands per column (low notes left), time vertical
                        let t = 2 * h - 1 - (2 * y + dy);
                        pix(t, 2 * x + dx)
                    };
                }
            }
            let (glyph, fg, bg) = quad_cell(px);
            if let Some(cell) = buf.cell_mut((area.x + x as u16, area.y + y as u16)) {
                cell.set_char(glyph)
                    .set_fg(Color::Rgb(fg.0, fg.1, fg.2))
                    .set_bg(Color::Rgb(bg.0, bg.1, bg.2));
            }
        }
    }
}

/// Note labels along the pitch axis, tinted their chromatone color.
fn draw_ruler(buf: &mut Buffer, area: Rect, app: &App) {
    let (lo, hi) = (app.params.range_lo, app.params.range_hi);
    let span = (hi - lo) as f32;
    let pixels = app.analyzer.num_bands() as f32;

    for c in (lo..=hi).filter(|n| n.rem_euclid(12) == 0) {
        // fractional band index of this C's center
        let i = ((c - lo) as f32 / span * pixels - 0.5).max(0.0);
        let (r, g, b) = note_color(c as f32);
        let color = Color::Rgb(r, g, b);
        let label = note_name(c);

        if app.params.horizontal {
            let row_px = pixels - 1.0 - i;
            let y = area.y + ((row_px / 2.0) as u16).min(area.height.saturating_sub(1));
            for (dx, ch) in label.chars().chain("╶".chars()).enumerate() {
                if let Some(cell) = buf.cell_mut((area.x + dx as u16, y)) {
                    cell.set_symbol(&ch.to_string()).set_fg(color);
                }
            }
        } else {
            // two bands per column in vertical mode
            let x = area.x + ((i / 2.0) as u16).min(area.width.saturating_sub(3));
            let y = area.y + area.height - 1;
            for (dx, ch) in label.chars().enumerate() {
                if let Some(cell) = buf.cell_mut((x + dx as u16, y)) {
                    cell.set_symbol(&ch.to_string()).set_fg(color);
                }
            }
        }
    }
}

/// Live tuner-style readout of the strongest detected pitch, top-right.
fn draw_readout(f: &mut Frame, area: Rect, app: &App) {
    let rect = Rect::new(area.right().saturating_sub(19), area.y + 1, 18, 4);
    let (title_color, lines) = match app.current_pitch {
        Some((note, freq, _v)) => {
            let nearest = note.round() as i32;
            let cents = ((note - nearest as f32) * 100.0).round() as i32;
            let (r, g, b) = note_color(note);
            (
                Color::Rgb(r, g, b),
                vec![
                    Line::from(Span::styled(
                        format!("{} {:+}¢", note_name(nearest), cents),
                        Style::default()
                            .fg(Color::Rgb(r, g, b))
                            .add_modifier(Modifier::BOLD),
                    )),
                    Line::from(format!("{freq:.1} Hz")),
                ],
            )
        }
        None => (
            Color::DarkGray,
            vec![Line::from("—"), Line::from("")],
        ),
    };
    f.render_widget(Clear, rect);
    f.render_widget(
        Paragraph::new(lines)
            .alignment(Alignment::Center)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(title_color)),
            )
            .style(Style::default().fg(Color::White).bg(Color::Black)),
        rect,
    );
}

fn draw_status(f: &mut Frame, area: Rect, app: &App) {
    let msg = if app.calibration.is_some() {
        Some(("measuring noise floor — stay quiet…", Color::Yellow))
    } else if app.paused {
        Some(("⏸ paused (space to resume)", Color::White))
    } else {
        None
    };
    if let Some((text, color)) = msg {
        let w = text.chars().count() as u16 + 2;
        let rect = Rect::new(area.x + (area.width.saturating_sub(w)) / 2, area.y, w, 1);
        f.render_widget(Clear, rect);
        f.render_widget(
            Paragraph::new(Line::from(text))
                .alignment(Alignment::Center)
                .style(Style::default().fg(color).bg(Color::Black)),
            rect,
        );
    }
}

/// Always-visible compact controls pane, left side.
fn draw_panel(f: &mut Frame, area: Rect, app: &App) {
    let height = (PARAMS.len() + 3) as u16;
    let rect = Rect::new(
        area.x + 2,
        area.y + area.height.saturating_sub(height) / 2,
        24.min(area.width),
        height.min(area.height),
    );
    let mut lines: Vec<Line> = vec![Line::from(Span::styled(
        " controls ",
        Style::default().add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
    ))];
    for (i, &param) in PARAMS.iter().enumerate() {
        let focused = i == app.focus;
        let marker = if focused { "▸" } else { " " };
        let style = if focused {
            Style::default()
                .fg(Color::Black)
                .bg(Color::White)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        };
        lines.push(Line::from(Span::styled(
            format!(
                "{marker}{} {:<9}{:>7} ",
                i + 1,
                param.label(),
                app.params.value_str(param)
            ),
            style,
        )));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        " ? help",
        Style::default().fg(Color::DarkGray),
    )));

    f.render_widget(Clear, rect);
    f.render_widget(
        Paragraph::new(lines).style(Style::default().bg(Color::Rgb(15, 15, 20))),
        rect,
    );
}

/// Centered help overlay (toggled with ?): keys plus what each setting does.
fn draw_help(f: &mut Frame, area: Rect, app: &App) {
    let mut lines: Vec<Line> = Vec::new();
    let key = Style::default().fg(Color::White).add_modifier(Modifier::BOLD);
    let dim = Style::default().fg(Color::Gray);
    let mut keyline = |k: &str, what: &str| {
        lines.push(Line::from(vec![
            Span::styled(format!("  {k:<16}"), key),
            Span::styled(what.to_string(), dim),
        ]));
    };
    keyline("↑/↓ · tab · 1-7", "select a control");
    keyline("←/→", "adjust it (shift = coarse steps)");
    keyline("c", "auto-calibrate sensitivity (stay quiet ~1s)");
    keyline("o", "flip orientation");
    keyline("space / enter", "pause / clear the screen");
    keyline("r", "reset all settings to defaults");
    keyline("? / esc", "close this help");
    keyline("q", "quit (settings are saved)");
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        "  what the settings do",
        Style::default().add_modifier(Modifier::BOLD | Modifier::UNDERLINED),
    )));
    for param in PARAMS {
        lines.push(Line::from(""));
        lines.push(Line::from(vec![
            Span::styled(format!("  {}: ", param.label()), key),
            Span::styled(param.explain().to_string(), dim),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!("  mic: {}", app.device_name()),
        Style::default().fg(Color::DarkGray),
    )));

    let width = 76.min(area.width.saturating_sub(4));
    // Rough height: wrapped explanation lines take ~2 rows each at this width.
    let height = (lines.len() as u16 + PARAMS.len() as u16 + 3).min(area.height);
    let rect = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + area.height.saturating_sub(height) / 2,
        width,
        height,
    );
    f.render_widget(Clear, rect);
    f.render_widget(
        Paragraph::new(lines)
            .wrap(ratatui::widgets::Wrap { trim: false })
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title(" chromaTUI help ")
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
            .style(Style::default().fg(Color::White).bg(Color::Rgb(15, 15, 20))),
        rect,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const RED: Rgb = (255, 0, 0);
    const BLACK: Rgb = (0, 0, 0);

    #[test]
    fn uniform_cell_is_full_block() {
        let (glyph, fg, _bg) = quad_cell([RED; 4]);
        assert_eq!(glyph, '█');
        assert_eq!(fg, RED);
    }

    #[test]
    fn top_bright_bottom_dark_is_upper_half() {
        let (glyph, fg, bg) = quad_cell([RED, RED, BLACK, BLACK]);
        assert_eq!(glyph, '▀');
        assert_eq!(fg, RED);
        assert_eq!(bg, BLACK);
    }

    #[test]
    fn single_bright_quadrants_pick_their_glyph() {
        assert_eq!(quad_cell([RED, BLACK, BLACK, BLACK]).0, '▘');
        assert_eq!(quad_cell([BLACK, RED, BLACK, BLACK]).0, '▝');
        assert_eq!(quad_cell([BLACK, BLACK, RED, BLACK]).0, '▖');
        assert_eq!(quad_cell([BLACK, BLACK, BLACK, RED]).0, '▗');
    }

    #[test]
    fn clusters_average_their_colors() {
        let dark_red = (60, 0, 0);
        let (glyph, fg, bg) = quad_cell([RED, (205, 0, 0), dark_red, (0, 0, 0)]);
        assert_eq!(glyph, '▀');
        assert_eq!(fg, (230, 0, 0)); // avg of the two bright reds
        assert_eq!(bg, (30, 0, 0)); // avg of the two dark pixels
    }
}

/// Confirmation modal for the reset action.
fn draw_reset_prompt(f: &mut Frame, area: Rect) {
    let text = "Reset all settings to defaults?";
    let width = (text.len() as u16 + 6).min(area.width);
    let rect = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + area.height.saturating_sub(5) / 2,
        width,
        5,
    );
    f.render_widget(Clear, rect);
    f.render_widget(
        Paragraph::new(vec![
            Line::from(""),
            Line::from(text),
            Line::from(Span::styled(
                "y: reset   n/esc: cancel",
                Style::default().fg(Color::Gray),
            )),
        ])
        .alignment(Alignment::Center)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow)),
        )
        .style(Style::default().fg(Color::White).bg(Color::Rgb(30, 25, 10))),
        rect,
    );
}
