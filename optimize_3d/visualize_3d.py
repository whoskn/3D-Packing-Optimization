import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from optimize_3d.constants import (
    VIS_BOX_ALPHA,
    VIS_FIG_SIZE,
    VIS_PACKAGE_ALPHA_MAX,
    VIS_PACKAGE_ALPHA_MIN,
    W_EXCEED_PENALTY,
    W_FLAT,
    W_FRAGILITY,
    W_GRAVITY,
    W_PYRAMID,
)
from optimize_3d.models import Box, PackedProduct


def _draw_objective_table(
    fig: plt.Figure,
    packed: list[PackedProduct],
    box: Box,
    colors,
    min_w: float,
    max_w: float,
    solver_obj: float | None = None,
) -> None:
    actual_bottom_faces = [p.dx * p.dy for p in packed]
    f_min = min(actual_bottom_faces)
    f_max = max(actual_bottom_faces)

    rows = []
    total_flat = total_pyramid = total_grav = total_pos = total_frag = 0.0
    for i, p in enumerate(packed):
        w_norm = (p.product.weight**2 - min_w**2) / (max_w**2 - min_w**2 or 1.0)
        dims = [p.product.width, p.product.length, p.product.depth]
        face_areas_i = [dims[a] * dims[b] for a, b in [(0, 1), (0, 2), (1, 2)]]
        min_face_i = min(face_areas_i)
        max_face_i = max(face_areas_i)
        face_diff = max_face_i / min_face_i
        face_norm = 1 - (p.dx * p.dy - min_face_i) / (max_face_i - min_face_i or 1.0)
        flat = W_FLAT * face_norm * face_diff
        face_norm_order = (p.dx * p.dy - f_min) / (f_max - f_min or 1.0)
        pyramid = W_PYRAMID * face_norm_order * p.z
        grav = W_GRAVITY * p.z * w_norm
        frag = (
            W_FRAGILITY
            / p.product.weight
            * sum(q.product.weight for q in packed if q is not p and q.z >= p.z + p.dz)
        )
        total_flat += flat
        total_pyramid += pyramid
        total_grav += grav
        total_frag += frag
        rows.append(
            [
                str(i),
                f"{p.product.weight:.1f}",
                f"{p.product.volume}",
                f"{flat:.1f}",
                f"{pyramid:.1f}",
                f"{grav:.1f}",
                f"{frag:.2f}",
            ]
        )

    max_x = max(p.x + p.dx for p in packed)
    max_y = max(p.y + p.dy for p in packed)
    max_z = max(p.z + p.dz for p in packed)
    exceed_x = max(0.0, max_x - box.x)
    exceed_y = max(0.0, max_y - box.y)
    exceed_z = max(0.0, max_z - box.z)
    exceed_term = (
        (W_EXCEED_PENALTY * exceed_x) ** 2
        + (W_EXCEED_PENALTY * exceed_y) ** 2
        + (W_EXCEED_PENALTY * exceed_z) ** 2
    )
    total_obj = (
        total_flat + total_pyramid + total_grav + total_pos + total_frag + exceed_term
    )

    rows.append(
        [
            "Σ",
            "",
            "",
            f"{total_flat:.1f}",
            f"{total_pyramid:.1f}",
            f"{total_grav:.1f}",
            f"{total_frag:.2f}",
        ]
    )

    col_labels = ["Pkg", "kg", "cm³", "flat", "pyra", "grav", "frag"]
    cell_colors = [
        [colors[i], "white", "white", "white", "white", "white", "white"]
        for i in range(len(packed))
    ]
    cell_colors.append(["#dddddd"] * 7)

    ax_tbl = fig.add_axes([0.61, 0.15, 0.38, 0.70])
    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="upper center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    obj_mismatch = solver_obj is not None and abs(total_obj - solver_obj) > 0.01
    title = f"Objective: {total_obj:.2f}"
    if solver_obj is not None:
        title += f"  (solver: {solver_obj:.2f})"
    title += f"\nxEx={exceed_x:.1f} yEx={exceed_y:.1f} zEx={exceed_z:.1f}cm  pen={exceed_term:.2f}"
    ax_tbl.set_title(
        title,
        fontsize=8,
        fontfamily="monospace",
        pad=4,
        color="red" if obj_mismatch else "black",
    )


def _visualize_single(
    packed: list[PackedProduct],
    box: Box | None,
    solver_obj: float | None = None,
    title: str | None = None,
) -> None:
    """Render a single carton packing in its own figure window."""
    fig = plt.figure(figsize=VIS_FIG_SIZE)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    if box is None:
        fig.text(
            0.3,
            0.02,
            "No suitable box available",
            fontsize=10,
            color="red",
            ha="center",
            fontweight="bold",
        )
    ax = fig.add_subplot(111, projection="3d")
    ax.set_position([0.0, 0.0, 0.60, 1.0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(packed)))

    def draw_box(origin, size, color, alpha=0.6):
        x, y, z = origin
        dx, dy, dz = size
        v = [
            [x, y, z],
            [x + dx, y, z],
            [x + dx, y + dy, z],
            [x, y + dy, z],
            [x, y, z + dz],
            [x + dx, y, z + dz],
            [x + dx, y + dy, z + dz],
            [x, y + dy, z + dz],
        ]
        faces = [
            [v[0], v[1], v[5], v[4]],
            [v[2], v[3], v[7], v[6]],
            [v[0], v[3], v[7], v[4]],
            [v[1], v[2], v[6], v[5]],
            [v[0], v[1], v[2], v[3]],
            [v[4], v[5], v[6], v[7]],
        ]
        ax.add_collection3d(
            Poly3DCollection(
                faces, alpha=alpha, facecolor=color, edgecolor="black", linewidth=0.5
            )
        )

    if box is None:
        render_box = Box(
            x=max(p.x + p.dx for p in packed),
            y=max(p.y + p.dy for p in packed),
            z=max(p.z + p.dz for p in packed),
        )
    else:
        render_box = box
        draw_box((0, 0, 0), (box.x, box.y, box.z), "lightgray", VIS_BOX_ALPHA)

    weights = [p.product.weight for p in packed]
    min_w, max_w = min(weights), max(weights)
    w_range = max_w - min_w if max_w > min_w else 1.0

    for i, p in enumerate(packed):
        t = (p.product.weight - min_w) / w_range
        alpha = VIS_PACKAGE_ALPHA_MIN + t * (
            VIS_PACKAGE_ALPHA_MAX - VIS_PACKAGE_ALPHA_MIN
        )
        draw_box((p.x, p.y, p.z), (p.dx, p.dy, p.dz), colors[i], alpha)
        ax.text(
            p.x + p.dx / 2,
            p.y + p.dy / 2,
            p.z + p.dz + 1,
            str(p.product.product_id),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            zorder=100,
        )

    ax.set_xlabel("Width")
    ax.set_ylabel("Depth")
    ax.set_zlabel("Height")
    ax.set_xlim(0, render_box.x)
    ax.set_ylim(0, render_box.y)
    ax.set_zlim(0, render_box.z)
    ax.set_box_aspect([render_box.x, render_box.y, render_box.z])

    _draw_objective_table(fig, packed, render_box, colors, min_w, max_w, solver_obj)


def visualize_packing(
    results: list[tuple[list[PackedProduct], Box | None, float]],
) -> None:
    """Visualize packing results — one figure window per carton."""
    n_cartons = len(results)
    for idx, (packed, box, solver_obj) in enumerate(results):
        title = f"Carton {idx + 1}/{n_cartons}" if n_cartons > 1 else None
        _visualize_single(packed, box, solver_obj, title)

    plt.show()
