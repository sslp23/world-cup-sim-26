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
    ('Germany',     'Scotland',       0.758, 0.242, 'Germany'),
    ('France',      'Sweden',         0.876, 0.124, 'France'),
    ('South Korea', 'Canada',         0.460, 0.540, 'Canada'),
    ('Netherlands', 'Morocco',        0.692, 0.308, 'Netherlands'),
    ('Colombia',    'Croatia',        0.462, 0.538, 'Croatia'),
    ('Spain',       'Algeria',        0.886, 0.114, 'Spain'),
    ('Paraguay',    'Austria',        0.585, 0.415, 'Paraguay'),
    ('Belgium',     'Czech Republic', 0.747, 0.253, 'Belgium'),
    ('Brazil',      'Japan',          0.725, 0.275, 'Brazil'),
    ('Ecuador',     'Norway',         0.498, 0.502, 'Norway'),
    ('Mexico',      'Ivory Coast',    0.665, 0.335, 'Mexico'),
    ('England',     'Senegal',        0.844, 0.156, 'England'),
    ('Argentina',   'Uruguay',        0.695, 0.305, 'Argentina'),
    ('Turkey',      'Iran',           0.662, 0.338, 'Turkey'),
    ('Switzerland', 'Egypt',          0.762, 0.238, 'Switzerland'),
    ('Portugal',    'Panama',         0.825, 0.175, 'Portugal'),
    # Round of 16 (indices 16-23)
    ('Germany',     'France',         0.300, 0.700, 'France'),
    ('Canada',      'Netherlands',    0.167, 0.833, 'Netherlands'),
    ('Croatia',     'Spain',          0.151, 0.849, 'Spain'),
    ('Paraguay',    'Belgium',        0.297, 0.703, 'Belgium'),
    ('Brazil',      'Norway',         0.584, 0.416, 'Brazil'),
    ('Mexico',      'England',        0.182, 0.818, 'England'),
    ('Argentina',   'Turkey',         0.787, 0.213, 'Argentina'),
    ('Switzerland', 'Portugal',       0.319, 0.681, 'Portugal'),
    # Quarter-finals (indices 24-27)
    ('France',      'Netherlands',    0.631, 0.369, 'France'),
    ('Spain',       'Belgium',        0.841, 0.159, 'Spain'),
    ('Brazil',      'England',        0.446, 0.554, 'England'),
    ('Argentina',   'Portugal',       0.687, 0.313, 'Argentina'),
    # Semi-finals (indices 28-29)
    ('France',      'Spain',          0.332, 0.668, 'Spain'),
    ('England',     'Argentina',      0.372, 0.628, 'Argentina'),
    # Final (index 30)
    ('Spain',       'Argentina',      0.648, 0.352, 'Spain'),
]

THIRD_PLACE = ('France', 'England', 0.620, 0.380, 'France')
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
