#!/usr/bin/env python3
"""
make_overview_gif.py -- animated "live demo" dashboard of SPEAR.

Pure matplotlib -> animated GIF (no co-sim, no heavy deps). One evolving
dashboard that walks through the loop at a glance:

  (1) SPEAR LLM planner prunes the attack menu 60 -> 6,
  (2) hierarchical RL (DQN target + SAC params) selects 6-of-10 systems,
  (3) power_disruption is injected into the grid digital twin,
  (4) the transmission-load and frequency dashboards plot live as the
      system deviates -- SPEAR's coordinated cascade vs a single-vector
      autonomous baseline.

The curves are an ILLUSTRATIVE overview calibrated to the paper's reported
endpoints (SPEAR ~ -59.4 MW / 0.21 Hz; autonomous ~ -10.1 MW / 0.03 Hz);
the quantitative results figures are produced by make_cascade_figures.py.

    python make_overview_gif.py        # writes figures/spear_overview.gif
"""
import os, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse
from PIL import Image

ASP = 11.0 / 6.2   # figure w/h; corrects circles drawn on the 0..100 bg axes

def acirc(ax, x, y, r, **kw):
    """Draw a visual circle on the aspect-'auto' background axes."""
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r * ASP, **kw))

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "figures"); os.makedirs(OUT, exist_ok=True)
GIF = os.path.join(OUT, "spear_overview.gif")

# ---- palette (matches the cascade figures) -------------------------------
INK   = "#1b2a3a"
MUTE  = "#9aa7b4"
BG    = "#f7f5f1"
PANEL = "#ffffff"
BEN   = "#4a4a4a"   # benign
SPE   = "#b23b3b"   # SPEAR
AUT   = "#2e6f95"   # autonomous
GRN   = "#2e7d5b"   # grid twin
plt.rcParams.update({"font.family": "serif", "savefig.facecolor": BG})

# ---- illustrative traces calibrated to the paper endpoints ---------------
Tt, START, DUR = 350.0, 140.0, 150.0
END = START + DUR
L0 = 300.0                      # benign transmission load baseline (MW)
A_L_S, A_L_A = 59.4, 10.1       # peak load disturbance (MW)
A_F_S, A_F_A = 0.21, 0.033      # peak |frequency deviation| (Hz)
N = 700
t = np.linspace(0.0, Tt, N)

def _dev(t, amp, tau_rise=34.0, tau_fall=95.0, osc=0.09, per=17.0):
    """Smooth disturbance: rise+saturate during attack, slow persistence after."""
    d = np.zeros_like(t)
    inw = (t >= START) & (t <= END)
    d[inw] = amp * (1 - np.exp(-(t[inw] - START) / tau_rise))
    aft = t > END
    tail = amp * (1 - np.exp(-(DUR) / tau_rise))
    d[aft] = tail * np.exp(-(t[aft] - END) / tau_fall)
    ripple = amp * osc * np.exp(-np.clip(t - START, 0, None) / 70.0) \
             * np.sin(2 * np.pi * np.clip(t - START, 0, None) / per)
    ripple[t < START] = 0.0
    return d + ripple

benign_L = L0 + 1.3 * np.sin(2 * np.pi * t / 73.0) + 0.6 * np.sin(2 * np.pi * t / 29.0)
benign_F = 0.004 * np.sin(2 * np.pi * t / 61.0)
spear_L  = benign_L - _dev(t, A_L_S)
auto_L   = benign_L - _dev(t, A_L_A, tau_fall=48.0)
spear_F  = benign_F - _dev(t, A_F_S, tau_rise=30.0, tau_fall=80.0)
auto_F   = benign_F - _dev(t, A_F_A, tau_rise=30.0, tau_fall=42.0)

# ---- animation schedule --------------------------------------------------
FRAMES = 56

def reveal_t(p):
    if p < 0.30:  return (p / 0.30) * (START * 0.97)
    if p < 0.88:  return START + ((p - 0.30) / 0.58) * (Tt - START)
    return Tt

def phase(p):
    if p < 0.15:  return 1
    if p < 0.30:  return 2
    if p < 0.88:  return 3
    return 4

def lit_targets(p):
    if p < 0.15:  return 0
    if p < 0.30:  return int(np.clip(((p - 0.15) / 0.15) * 6 + 0.5, 0, 6))
    return 6

STAGES = [
    ("SPEAR  ·  LLM planner", "prune 60 → 6 CVE-anchored vectors", SPE),
    ("Hierarchical RL", "DQN target  +  SAC magnitude / stealth", INK),
    ("Grid digital twin", "10 DS · IEEE-34  +  multi-layer IDS", GRN),
]

def status_text(ph, lit):
    if ph == 1: return "SPEAR LLM planner  —  pruning the attack menu  60 → 6 CVE-anchored vectors"
    if ph == 2: return f"Hierarchical RL  —  DQN selecting targets  ({lit}/6 of 10 DS)  ·  SAC tuning magnitude"
    if ph == 3: return "Grid digital twin  —  power_disruption injected  ·  coordinated cascade propagating"
    return "System deviated  —  coordinated 6-of-10 cascade:   −59.4 MW load   ·   0.21 Hz frequency"

# ---- drawing -------------------------------------------------------------
def box(ax, x, y, w, h, title, sub, color, active):
    fc = color if active else PANEL
    tc = "white" if active else MUTE
    ec = color if active else "#d9d2c7"
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.5,rounding_size=1.8",
                 fc=fc, ec=ec, lw=2.2 if active else 1.3, zorder=3))
    ax.text(x, y + 2.1, title, ha="center", va="center", color=tc,
            fontsize=11.5, fontweight="bold", zorder=4)
    ax.text(x, y - 2.6, sub, ha="center", va="center",
            color="white" if active else MUTE, fontsize=8.6, zorder=4)

def draw_plot(ax, rv, yb, ys, ya, ylabel, ylim, callout):
    k = int(np.searchsorted(t, rv))
    k = max(2, min(k, N))
    ax.set_facecolor(PANEL)
    ax.axvspan(START, END, color=SPE, alpha=0.05, zorder=0)
    ax.plot(t[:k], yb[:k], color=BEN, lw=2.0, label="benign", zorder=2)
    ax.plot(t[:k], ya[:k], color=AUT, lw=1.8, ls=(0, (4, 2)), label="autonomous", zorder=2)
    ax.plot(t[:k], ys[:k], color=SPE, lw=2.6, label="SPEAR", zorder=3)
    ax.axvline(rv, color=INK, lw=1.2, alpha=0.55, zorder=4)          # sweeping "now" cursor
    ax.scatter([rv], [ys[min(k-1, N-1)]], s=26, color=SPE, zorder=5, edgecolor="white", linewidth=0.8)
    ax.set_xlim(0, Tt); ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel, fontsize=11.5)
    ax.tick_params(labelsize=9.5)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9, ncol=3,
              handlelength=1.4, columnspacing=1.1, borderpad=0.35)
    if callout and rv > END - 8:
        yv = ys[min(k-1, N-1)]
        ax.annotate(callout, xy=(rv, yv), xytext=(rv - 96, yv + (ylim[1]-ylim[0])*0.17),
                    fontsize=10.5, fontweight="bold", color=SPE,
                    arrowprops=dict(arrowstyle="->", color=SPE, lw=1.4))

def render_frame(p):
    ph, lit, rv = phase(p), lit_targets(p), reveal_t(p)
    fig = plt.figure(figsize=(11.0, 6.2), dpi=100)
    fig.patch.set_facecolor(BG)
    bg = fig.add_axes([0, 0, 1, 1]); bg.set_xlim(0, 100); bg.set_ylim(0, 100); bg.axis("off")

    # header
    bg.text(2.5, 95.5, "SPEAR", ha="left", va="center", fontsize=23, fontweight="bold", color=INK)
    bg.text(16.5, 95.8, "Semantic Planning & Execution of Adversarial Red-teaming",
            ha="left", va="center", fontsize=11, color=SPE, style="italic")
    bg.text(2.5, 90.3, "live attack demonstration  ·  LLM-guided RL  →  EVCS-integrated grid",
            ha="left", va="center", fontsize=10, color="#6b6b6b")

    # ---- left column: pipeline stages ----
    cx = 15.5
    ys_box = [80, 65, 50]
    for i, ((ti, su, co), yb) in enumerate(zip(STAGES, ys_box)):
        active = (ph == i + 1) or (ph == 4 and i == 2)
        box(bg, cx, yb, 25, 10.5, ti, su, co, active)
        if i < 2:
            on = (ph >= i + 2) or ph == 4
            bg.add_patch(FancyArrowPatch((cx, yb - 5.4), (cx, ys_box[i+1] + 5.4),
                         arrowstyle="-|>", mutation_scale=14, lw=2.0,
                         color=INK if on else "#d2cabc", zorder=2))
    # feedback arrow (twin -> RL)
    bg.add_patch(FancyArrowPatch((cx + 13, 50), (cx + 13, 65),
                 connectionstyle="arc3,rad=-0.55", arrowstyle="-|>",
                 mutation_scale=12, lw=1.5, ls="--", color="#b9b0a2", zorder=1))
    bg.text(cx + 15.6, 57.5, "RL\nfeedback", ha="left", va="center", fontsize=7.6, color="#8a8378")

    # ---- target grid (10 DS, up to 6 attacked) ----
    bg.text(cx, 40.0, "10 distribution systems", ha="center", va="center",
            fontsize=9.5, color=INK, fontweight="bold")
    gx0, gy0, dx, dy = 6.5, 33.0, 4.4, 6.2
    for idx in range(10):
        r, c = divmod(idx, 5)
        x = gx0 + c * dx; y = gy0 - r * dy
        hit = idx < lit
        if hit:
            acirc(bg, x, y, 2.3, fc="none", ec=SPE, lw=1.0, alpha=0.4, zorder=2)
        acirc(bg, x, y, 1.45, fc=SPE if hit else PANEL,
              ec=SPE if hit else MUTE, lw=1.6, zorder=3)
    bg.text(cx, 22.0, f"{lit} of 10 attacked  ·  ≤6 budget", ha="center", va="center",
            fontsize=9, color=SPE if lit else MUTE)

    # ---- right: live dashboards ----
    ax_load = fig.add_axes([0.42, 0.57, 0.545, 0.33])
    ax_freq = fig.add_axes([0.42, 0.155, 0.545, 0.33])
    draw_plot(ax_load, rv, benign_L, spear_L, auto_L, "transmission load (MW)",
              (228, 312), "−59.4 MW")
    draw_plot(ax_freq, rv, benign_F, spear_F, auto_F, "frequency Δf (Hz)",
              (-0.255, 0.055), "0.21 Hz peak")
    ax_freq.set_xlabel("time (s)", fontsize=11.5)

    # ---- full-width status line at the very bottom ----
    bg.add_patch(FancyBboxPatch((3, 2.2), 94, 5.6, boxstyle="round,pad=0.3,rounding_size=1.2",
                 fc=INK, ec="none", zorder=3))
    dot = {1: SPE, 2: INK, 3: GRN, 4: SPE}[ph]
    acirc(bg, 6.0, 5.0, 1.05, fc=dot, ec="white", lw=1.0, zorder=4)
    bg.text(9.0, 5.0, status_text(ph, lit), ha="left", va="center",
            color="white", fontsize=10.5, zorder=4)

    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor=BG); plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

# ---- build the gif -------------------------------------------------------
if __name__ == "__main__":
    frames = [render_frame(k / (FRAMES - 1)) for k in range(FRAMES)]
    # quantize to keep the file small
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    durs = [95] * FRAMES
    durs[0] = 900
    durs[-1] = 2600
    pal[0].save(GIF, save_all=True, append_images=pal[1:], duration=durs,
                loop=0, optimize=True, disposal=2)
    sz = os.path.getsize(GIF) / 1024
    print(f"wrote {GIF}  ({FRAMES} frames, {sz:.0f} KB)")
