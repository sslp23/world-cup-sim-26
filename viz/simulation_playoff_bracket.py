"""
WC 2026 predicted playoff bracket — radial NetworkX tree.

31-node balanced binary tree (5 rounds: Round of 32 → Final).
Match order is arranged so the tree edges reflect the actual bracket
flow (winner of each match feeds the correct parent node).

Data: simulation/output/wc_26_catboost.xlsx

Run from the project root:
    python -m viz.simulation_playoff_bracket
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

OUTPUT_PATH = 'viz/output/simulation_playoff_bracket.png'

# ── Match data ────────────────────────────────────────────────────────────────
# Ordered so that reversed flat list maps correctly onto the balanced tree.
# Each tuple: (team_a, team_b, p_a, p_b, winner)
ROUNDS = [
    # Round of 32 (indices 0-15) — pairs whose winners meet in the R16 match above
    ('Ecuador',       'United States', 0.654, 0.346, 'Ecuador'),
    ('France',        'Sweden',        0.736, 0.264, 'France'),
    ('Czech Republic','Canada',        0.428, 0.572, 'Canada'),
    ('Netherlands',   'Morocco',       0.543, 0.457, 'Netherlands'),
    ('Colombia',      'Croatia',       0.503, 0.497, 'Colombia'),
    ('Spain',         'Austria',       0.756, 0.244, 'Spain'),
    ('Turkey',        'Algeria',       0.603, 0.397, 'Turkey'),
    ('Belgium',       'South Korea',   0.734, 0.266, 'Belgium'),
    ('Brazil',        'Japan',         0.614, 0.386, 'Brazil'),
    ('Germany',       'Senegal',       0.598, 0.402, 'Germany'),
    ('Mexico',        'Scotland',      0.602, 0.398, 'Mexico'),
    ('England',       'Norway',        0.627, 0.373, 'England'),
    ('Argentina',     'Uruguay',       0.612, 0.388, 'Argentina'),
    ('Paraguay',      'Iran',          0.618, 0.382, 'Paraguay'),
    ('Switzerland',   'Egypt',         0.581, 0.419, 'Switzerland'),
    ('Portugal',      'Ivory Coast',   0.714, 0.286, 'Portugal'),
    # Round of 16 (indices 16-23)
    ('Ecuador',       'France',        0.412, 0.588, 'France'),
    ('Canada',        'Netherlands',   0.273, 0.727, 'Netherlands'),
    ('Colombia',      'Spain',         0.385, 0.615, 'Spain'),
    ('Turkey',        'Belgium',       0.366, 0.634, 'Belgium'),
    ('Brazil',        'Germany',       0.583, 0.417, 'Brazil'),
    ('Mexico',        'England',       0.278, 0.722, 'England'),
    ('Argentina',     'Paraguay',      0.691, 0.309, 'Argentina'),
    ('Switzerland',   'Portugal',      0.334, 0.666, 'Portugal'),
    # Quarter-finals (indices 24-27)
    ('France',        'Netherlands',   0.576, 0.424, 'France'),
    ('Spain',         'Belgium',       0.700, 0.300, 'Spain'),
    ('Brazil',        'England',       0.485, 0.515, 'England'),
    ('Argentina',     'Portugal',      0.565, 0.435, 'Argentina'),
    # Semi-finals (indices 28-29)
    ('France',        'Spain',         0.403, 0.597, 'Spain'),
    ('England',       'Argentina',     0.464, 0.536, 'Argentina'),
    # Final (index 30)
    ('Spain',         'Argentina',     0.584, 0.416, 'Spain'),
]

THIRD_PLACE = ('France', 'England', 0.467, 0.533, 'England')
CHAMPION    = 'Spain'

ROUND_NAMES = ['Round of 32', 'Round of 16', 'Quarter-Final', 'Semi-Final', 'Final']

C_SPINE = '#CFD8DC'
C_ANNOT = '#546E7A'
C_EDGE  = '#A5D6A7'


def make_label(ta, tb, pa, pb, winner):
    def abbrev(team):
        return team if len(team) <= 11 else team[:10] + '.'
    sa = '★' if winner == ta else '  '
    sb = '★' if winner == tb else '  '
    return f'{sa}{abbrev(ta)}({pa:.2f})\n{sb}{abbrev(tb)}({pb:.2f})'


def radial_layout(G, r_step=230):
    """
    Recursive radial layout — two halves spread LEFT and RIGHT from root.
    Children at each level share equal angular wedges within their parent's wedge.
    """
    pos = {}

    def kids(node, parent):
        return [n for n in G.neighbors(node) if n != parent]

    def place(node, parent, radius, lo, hi):
        mid = (lo + hi) / 2
        pos[node] = (radius * np.cos(mid), radius * np.sin(mid))
        ch = kids(node, parent)
        if not ch:
            return
        w = (hi - lo) / len(ch)
        for i, c in enumerate(ch):
            place(c, node, radius + r_step, lo + i * w, lo + (i + 1) * w)

    pos[0] = (0.0, 0.0)
    rk   = list(G.neighbors(0))
    span = 2 * np.pi / len(rk)
    # Start at -π/2 so the two halves go RIGHT and LEFT
    start = -np.pi / 2
    for i, c in enumerate(rk):
        place(c, 0, r_step, start + i * span, start + (i + 1) * span)
    return pos


def run():
    G = nx.balanced_tree(2, 4)

    labels_flat = [make_label(*r) for r in ROUNDS]
    labels_rev  = list(reversed(labels_flat))
    labels_dict = {n: labels_rev[n] for n in G.nodes()}

    depth_map   = nx.single_source_shortest_path_length(G, 0)
    cmap        = plt.cm.Greens
    node_colors = [cmap(0.95 - depth_map[n] * 0.17) for n in G.nodes()]
    node_sizes  = [1500 - depth_map[n] * 200 for n in G.nodes()]

    pos = radial_layout(G)

    # Push label anchor outward from centre so it clears the node circle
    cx, cy = 0.0, 0.0
    label_pos = {}
    for n, (x, y) in pos.items():
        dx, dy = x - cx, y - cy
        dist    = max(np.hypot(dx, dy), 1.0)
        push    = 14 + depth_map[n] * 9
        label_pos[n] = (x + dx / dist * push, y + dy / dist * push)
    # Root: push label downward so it sits below the node
    label_pos[0] = (cx, cy - 40)

    fig, ax = plt.subplots(figsize=(24, 18))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    nx.draw_networkx(
        G, pos=pos, ax=ax,
        with_labels=False,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color=C_EDGE,
        width=2.8,
    )
    nx.draw_networkx_labels(
        G, pos=label_pos, ax=ax,
        labels=labels_dict,
        font_size=6.8,
        font_family='monospace',
        bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#90A4AE',
                  lw=0.6, alpha=0.97),
    )

    # ── Round labels: vertical column on the north axis (x≈0) ──────────────
    # Each label is placed 85 units above the topmost node of its ring.
    # The north axis is the empty gap between the left and right halves.
    depth_max_y = {}
    for n, (x, y) in pos.items():
        d = depth_map[n]
        depth_max_y[d] = max(depth_max_y.get(d, -1e9), y)

    depth_min_y = {n: min(y for x, y in pts) for n, pts in
                   {d: [(x, y) for n, (x, y) in pos.items() if depth_map[n] == d]
                    for d in range(5)}.items()}

    for d in range(1, 5):   # skip d=0, handled by Final badge + champion banner
        # depth 1 = Semi-Final, depth 4 = Round of 32 → reverse index
        rname    = ROUND_NAMES[4 - d]
        bg_color = cmap(0.95 - d * 0.17)
        txt_col  = 'white' if d <= 2 else '#1A1A1A'
        # North label (above topmost node of ring)
        ax.text(0, depth_max_y[d] + 85, rname,
                ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                color=txt_col,
                bbox=dict(boxstyle='round,pad=0.35', fc=bg_color,
                          ec='none', alpha=0.92))
        # South label (below bottommost node of ring)
        ax.text(0, depth_min_y[d] - 85, rname,
                ha='center', va='top', fontsize=8.5, fontweight='bold',
                color=txt_col,
                bbox=dict(boxstyle='round,pad=0.35', fc=bg_color,
                          ec='none', alpha=0.92))

    # ── Final label + champion banner ────────────────────────────────────────
    ax.text(cx, cy + 38, 'Final',
            ha='center', va='bottom', fontsize=8.5, fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', fc=cmap(0.95), ec='none', alpha=0.90))
    ax.text(cx, cy - 42, f'★  CHAMPION:  {CHAMPION}',
            ha='center', va='top', fontsize=11, fontweight='bold',
            color='#1B5E20',
            bbox=dict(boxstyle='round,pad=0.45', fc='#F9FBE7',
                      ec='#558B2F', lw=1.2))

    # ── 3rd-place note ───────────────────────────────────────────────────────
    ta, tb, pa, pb, win = THIRD_PLACE
    ax.text(0.5, 0.01,
            f'3rd-place play-off  ·  {ta} ({pa:.2f}) vs {tb} ({pb:.2f})  →  ★ {win}',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=9, color=C_ANNOT, fontstyle='italic')

    ax.set_title(
        'WC 2026 — Predicted Playoff Bracket  (CatBoost model)',
        fontsize=14, fontweight='bold', color=C_ANNOT, pad=12,
    )
    ax.axis('equal')
    ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUTPUT_PATH}')
    plt.show()


if __name__ == '__main__':
    run()
