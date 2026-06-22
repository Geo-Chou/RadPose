import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


OUTPUT_PATH = Path('work_dirs/figures/spherical_kernel_branches.png')


def polar_sector(r0, r1, a0, a1, n=32):
    outer = [
        (r1 * math.cos(t), r1 * math.sin(t))
        for t in [a0 + (a1 - a0) * i / n for i in range(n + 1)]
    ]
    inner = [
        (r0 * math.cos(t), r0 * math.sin(t))
        for t in [a1 - (a1 - a0) * i / n for i in range(n + 1)]
    ]
    return outer + inner


def draw_grid(ax, radii, angles, cmap='viridis'):
    total = len(radii) - 1
    for ridx in range(total):
        for aidx in range(len(angles) - 1):
            patch = Polygon(
                polar_sector(radii[ridx], radii[ridx + 1], angles[aidx], angles[aidx + 1]),
                closed=True,
                facecolor='white',
                edgecolor='black',
                linewidth=1.0,
                alpha=1.0)
            ax.add_patch(patch)


def highlight_cells(ax, radii, angles, cells, color='none', edgecolor='#ff8c00', linewidth=2.8):
    for ridx, aidx in cells:
        patch = Polygon(
            polar_sector(radii[ridx], radii[ridx + 1], angles[aidx], angles[aidx + 1]),
            closed=True,
            facecolor=color,
            edgecolor=edgecolor,
            linewidth=linewidth)
        ax.add_patch(patch)


def format_ax(ax, title):
    ax.set_title(title, fontsize=14)
    ax.set_aspect('equal')
    ax.set_xlim(-0.25, 6.8)
    ax.set_ylim(-3.1, 3.1)
    ax.set_xlabel('x (meters)')
    ax.set_ylabel('y (meters)')


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    radii = [1.0, 2.0, 3.4, 4.8, 6.2]
    angles = [math.radians(v) for v in (-25, -12.5, 0, 12.5, 25)]
    center = (1, 1)

    conventional = [
        (r, a)
        for r in range(center[0] - 1, center[0] + 2)
        for a in range(center[1] - 1, center[1] + 2)
        if 0 <= r < len(radii) - 1 and 0 <= a < len(angles) - 1
    ]
    radial = [
        (r, center[1])
        for r in range(center[0] - 1, center[0] + 2)
        if 0 <= r < len(radii) - 1
    ]
    angular = [
        (center[0], a)
        for a in range(center[1] - 1, center[1] + 2)
        if 0 <= a < len(angles) - 1
    ]
    pointwise = [center]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    panels = [
        ('Conventional 3D Conv', conventional, '#ffb347'),
        ('Radial Branch', radial, '#ffb347'),
        ('Angular Branch', angular, '#ffb347'),
        ('Point-wise Branch', pointwise, '#ffb347'),
    ]

    for ax, (title, cells, color) in zip(axes, panels):
        draw_grid(ax, radii, angles)
        highlight_cells(ax, radii, angles, cells, color=color)
        format_ax(ax, title)

    fig.suptitle('Spherical Receptive Fields on the R-A Plane', fontsize=16)
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches='tight')
    print(OUTPUT_PATH)


if __name__ == '__main__':
    main()
