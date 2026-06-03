"""
ML-Poisson two-stage architecture flow diagram.

Stage 1 — XGBoost goal regressor → (λ_home, λ_away)
Stage 2 — Dixon-Coles score matrix → P(home win), P(draw), P(away win)

Run from the project root:
    python -m viz.model_ml_poisson_architecture
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT_PATH = 'viz/output/model_ml_poisson_architecture.png'

C_FEAT   = '#455A64'
C_XGB    = '#1F4E79'
C_LAM    = '#2E75B6'
C_DC     = '#0D47A1'
C_HW     = '#1F4E79'
C_DRAW   = '#607D8B'
C_AW     = '#90A4AE'
C_ANNOT  = '#37474F'
C_ARROW  = '#546E7A'
C_BORDER = '#B0BEC5'


def fbox(ax, cx, cy, w, h, fc, ec=None, zorder=3):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle='round,pad=0.01',
        facecolor=fc, edgecolor=ec or C_BORDER,
        linewidth=0.8, zorder=zorder,
    )
    ax.add_patch(patch)


def txt(ax, x, y, text, fs=9, fw='normal', color='white',
        ha='center', va='center', style='normal', zorder=5):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, fontweight=fw,
            color=color, fontstyle=style, zorder=zorder)


def arr(ax, x1, y1, x2, y2):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->', color=C_ARROW, lw=1.5,
            connectionstyle='arc3,rad=0.0',
        ), zorder=2,
    )


def run():
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 9)
    ax.axis('off')

    CX = 5.5   # horizontal center

    # ── Stage labels (left margin) ────────────────────────────────────────────
    for y, stage_lbl in [(8.1, 'INPUT'), (6.5, 'STAGE 1'), (3.3, 'STAGE 2'), (1.2, 'OUTPUT')]:
        ax.text(0.15, y, stage_lbl, ha='left', va='center',
                fontsize=7, fontweight='bold', color='#90A4AE',
                rotation=90, zorder=5)

    # ── Features box ──────────────────────────────────────────────────────────
    fbox(ax, CX, 8.1, w=8.8, h=0.78, fc=C_FEAT, ec='#546E7A')
    txt(ax, CX, 8.28, 'Match Features', fs=10.5, fw='bold')
    txt(ax, CX, 8.02,
        'ELO diff  ·  WC history (best round, goals/game, games)  ·  Form MAs (attack & defense, last 5 & 20)',
        fs=7.8, style='italic')

    # ── Arrow: Features → XGBoost ─────────────────────────────────────────────
    arr(ax, CX, 7.71, CX, 7.10)

    # ── XGBoost box ───────────────────────────────────────────────────────────
    fbox(ax, CX, 6.5, w=7.8, h=1.12, fc=C_XGB)
    txt(ax, CX, 6.76, 'XGBoost Goal Regressor', fs=11.5, fw='bold')
    txt(ax, CX, 6.50, 'Single model  ·  Poisson objective  (count:poisson)', fs=8.5, style='italic')
    txt(ax, CX, 6.27, 'Trained pooling home & away attacker views  →  2× training data', fs=7.5, style='italic')

    # ── Arrows: XGBoost → λ boxes ─────────────────────────────────────────────
    arr(ax, 4.7, 5.94, 3.2, 5.38)   # → λ_home
    arr(ax, 6.3, 5.94, 7.8, 5.38)   # → λ_away

    # ── Lambda boxes ──────────────────────────────────────────────────────────
    for cx, title, sub in [
        (3.2, 'λ_home', 'E [home goals]'),
        (7.8, 'λ_away', 'E [away goals]'),
    ]:
        fbox(ax, cx, 5.05, w=2.3, h=0.64, fc=C_LAM)
        txt(ax, cx, 5.20, title, fs=13, fw='bold')
        txt(ax, cx, 4.91, sub, fs=8, style='italic')

    # ── Arrows: λ → Dixon-Coles ───────────────────────────────────────────────
    arr(ax, 3.2, 4.73, 3.8, 4.01)
    arr(ax, 7.8, 4.73, 7.2, 4.01)

    # ── Dixon-Coles box ───────────────────────────────────────────────────────
    fbox(ax, CX, 3.3, w=8.8, h=1.42, fc=C_DC)
    txt(ax, CX, 3.76, 'Dixon-Coles Score Matrix', fs=11.5, fw='bold')
    txt(ax, CX, 3.52,
        'P(i, j)  =  τ(i, j)  ·  Pois(i | λ_home)  ·  Pois(j | λ_away)',
        fs=9.5, style='italic')
    txt(ax, CX, 3.29,
        'τ  corrects Poisson underestimation of low-score results  {0–0, 1–0, 0–1, 1–1}',
        fs=7.8, style='italic')
    txt(ax, CX, 3.06,
        'Symmetrized:  (M_forward + M_inverted.T) / 2  →  removes label bias at neutral venues',
        fs=7.5, style='italic')

    # ── Mini score-matrix illustration (inside DC box, right side) ────────────
    # 5×5 grid showing lower-tri = blue, diagonal = grey, upper-tri = light
    MX0, MY0 = 8.95, 3.72   # top-left corner of mini grid
    cell_w, cell_h = 0.21, 0.19
    N = 5
    for i in range(N):
        for j in range(N):
            if i > j:
                fc = '#4A90C4'    # lower triangle — home win
            elif i == j:
                fc = '#78909C'    # diagonal — draw
            else:
                fc = '#B0BEC5'    # upper triangle — away win
            rect = plt.Rectangle(
                (MX0 + j * cell_w, MY0 - (i + 1) * cell_h),
                cell_w, cell_h,
                facecolor=fc, edgecolor='white', linewidth=0.4, zorder=4,
            )
            ax.add_patch(rect)
    # Axis labels for the mini grid
    txt(ax, MX0 + N * cell_w / 2, MY0 + 0.08, 'away goals →', fs=6, color='#B0D0EE')
    ax.text(MX0 - 0.08, MY0 - N * cell_h / 2, '← home', va='center',
            fontsize=6, color='#B0D0EE', rotation=90, zorder=5)

    # ── Arrows: DC → outcomes ─────────────────────────────────────────────────
    arr(ax, 3.5, 2.59, 2.0, 1.60)   # → P(Home Win)
    arr(ax, CX,  2.59, CX,  1.60)   # → P(Draw)
    arr(ax, 7.5, 2.59, 9.0, 1.60)   # → P(Away Win)

    # ── Outcome boxes ─────────────────────────────────────────────────────────
    for cx, fc, title, sub in [
        (2.0, C_HW,   'P(Home Win)',  'Σ lower triangle'),
        (CX,  C_DRAW, 'P(Draw)',      'Σ diagonal'),
        (9.0, C_AW,   'P(Away Win)',  'Σ upper triangle'),
    ]:
        fbox(ax, cx, 1.2, w=2.9, h=0.74, fc=fc)
        txt(ax, cx, 1.34, title, fs=10, fw='bold')
        txt(ax, cx, 1.07, sub, fs=8, style='italic')

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.975, 'ML-Poisson Hybrid Model — Architecture',
             ha='center', va='top', fontsize=14, fontweight='bold', color=C_ANNOT)

    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUTPUT_PATH}')
    plt.show()


if __name__ == '__main__':
    run()
