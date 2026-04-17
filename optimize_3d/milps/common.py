from itertools import permutations

import gurobipy as gp
from gurobipy import GRB

from optimize_3d.constants import CUBENESS_THRESHOLD
from optimize_3d.models import Box, PackedProduct, Product

ROTATION_DIMS = tuple(permutations([0, 1, 2]))


def _create_position_vars(
    model: gp.Model,
    data: list[Product],
    box: Box,
    n_range: range,
    warm_start: list[PackedProduct] | None = None,
) -> tuple[
    list[gp.Var],
    list[gp.Var],
    list[gp.Var],
    dict[int, gp.Var],
    dict[int, gp.Var],
    dict[int, gp.Var],
]:
    """Create continuous position vars (xs, ys, zs) and integer dimension
    vars (dx, dy, dz) for every product.

    When *warm_start* is provided the position variables are initialised
    from the previous solution; otherwise the first product is placed at the
    origin.
    """
    xs, ys, zs, dxs, dys, dzs = [], [], [], [], [], []
    for i in n_range:
        min_dim = min(data[i].length, data[i].width, data[i].depth)
        x = model.addVar(
            name=f"xs_{i}",
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=box.x - min_dim,
        )
        y = model.addVar(
            name=f"ys_{i}",
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=box.y - min_dim,
        )
        z = model.addVar(
            name=f"zs_{i}",
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=box.z - min_dim,
        )
        dx = model.addVar(name=f"dx_{i}", vtype=GRB.INTEGER)
        dy = model.addVar(name=f"dy_{i}", vtype=GRB.INTEGER)
        dz = model.addVar(name=f"dz_{i}", vtype=GRB.INTEGER)

        if warm_start:
            x.Start = warm_start[i].x
            y.Start = warm_start[i].y
            z.Start = warm_start[i].z
            dx.Start = warm_start[i].dx
            dy.Start = warm_start[i].dy
            dz.Start = warm_start[i].dz

        xs.append(x)
        ys.append(y)
        zs.append(z)
        dxs.append(dx)
        dys.append(dy)
        dzs.append(dz)

    return xs, ys, zs, dxs, dys, dzs


def _add_rotation_constraints(
    model: gp.Model,
    data: list[Product],
    n_range: range,
    dx: dict[int, gp.Var],
    dy: dict[int, gp.Var],
    dz: dict[int, gp.Var],
) -> list[list[gp.Var]]:
    """Add rotation binary variables, cubeness-based pruning, and constraints
    linking dx/dy/dz to the chosen rotation.

    Returns the list of rotation variable lists (one per product).
    """
    cl, cu = 1 - CUBENESS_THRESHOLD, 1 + CUBENESS_THRESHOLD
    rotations = []

    for i in n_range:
        rotation = [
            model.addVar(name=f"rotation_{i}_{d}", vtype=GRB.BINARY) for d in range(6)
        ]
        rotations.append(rotation)

        # Prune equivalent rotations for near-cubic dimensions
        wd, ln, dp = data[i].width, data[i].length, data[i].depth
        wl_similar = cl < wd / ln < cu
        dl_similar = cl < dp / ln < cu
        wd_similar = cl < wd / dp < cu

        if wl_similar and dl_similar and wd_similar:
            model.addConstr(rotation[0] == 1)
        elif wl_similar:
            for r in (2, 3, 5):
                model.addConstr(rotation[r] == 0)
        elif dl_similar:
            for r in (1, 4, 5):
                model.addConstr(rotation[r] == 0)
        elif wd_similar:
            for r in (3, 4, 5):
                model.addConstr(rotation[r] == 0)

        model.addConstr(gp.quicksum(rotation) == 1)

        # Link rotated dimensions to dx, dy, dz
        dims = [data[i].width, data[i].length, data[i].depth]
        model.addConstr(
            dx[i]
            == gp.quicksum(rotation[r] * dims[ROTATION_DIMS[r][0]] for r in range(6))
        )
        model.addConstr(
            dy[i]
            == gp.quicksum(rotation[r] * dims[ROTATION_DIMS[r][1]] for r in range(6))
        )
        model.addConstr(
            dz[i]
            == gp.quicksum(rotation[r] * dims[ROTATION_DIMS[r][2]] for r in range(6))
        )

    return rotations


def _add_separation_constraints(
    model: gp.Model,
    xs: list[gp.Var],
    ys: list[gp.Var],
    zs: list[gp.Var],
    dx: dict[int, gp.Var],
    dy: dict[int, gp.Var],
    dz: dict[int, gp.Var],
    n_products: int,
    box: Box,
) -> None:
    """Add pairwise big-M non-overlap constraints for all product pairs.

    Uses axis-specific big-M values (box.x/y/z) instead of a single global M,
    which tightens the LP relaxation and reduces branch-and-bound nodes.
    """
    Mx, My, Mz = box.x, box.y, box.z
    for i in range(n_products):
        for j in range(i + 1, n_products):
            sep = [
                model.addVar(name=f"s{d}_{i}_{j}", vtype=GRB.BINARY) for d in range(6)
            ]
            model.addConstr(gp.quicksum(sep) >= 1)
            model.addConstr(xs[i] + dx[i] <= xs[j] + Mx * (1 - sep[0]))
            model.addConstr(xs[j] + dx[j] <= xs[i] + Mx * (1 - sep[1]))
            model.addConstr(ys[i] + dy[i] <= ys[j] + My * (1 - sep[2]))
            model.addConstr(ys[j] + dy[j] <= ys[i] + My * (1 - sep[3]))
            model.addConstr(zs[i] + dz[i] <= zs[j] + Mz * (1 - sep[4]))
            model.addConstr(zs[j] + dz[j] <= zs[i] + Mz * (1 - sep[5]))


def _add_bounding_box_vars(
    model: gp.Model,
    xs: list[gp.Var],
    ys: list[gp.Var],
    zs: list[gp.Var],
    dx: dict[int, gp.Var],
    dy: dict[int, gp.Var],
    dz: dict[int, gp.Var],
    n_range: range,
    vtype: str = GRB.CONTINUOUS,
) -> tuple[gp.Var, gp.Var, gp.Var]:
    """Add max_x, max_y, max_z variables constrained to be >= every product's
    upper bound on each axis. Returns (max_x, max_y, max_z)."""
    max_x = model.addVar(name="max_x", lb=0, vtype=vtype)
    max_y = model.addVar(name="max_y", lb=0, vtype=vtype)
    max_z = model.addVar(name="max_z", lb=0, vtype=vtype)

    for i in n_range:
        model.addConstr(max_x >= xs[i] + dx[i])
        model.addConstr(max_y >= ys[i] + dy[i])
        model.addConstr(max_z >= zs[i] + dz[i])

    return max_x, max_y, max_z


def pull_to_origin(packed: list) -> list:
    """Pull every packed product towards (0,0,0) as far as possible
    without overlapping other products or leaving the box."""

    def overlaps(a, b):
        return not (
            a.x + a.dx <= b.x
            or b.x + b.dx <= a.x
            or a.y + a.dy <= b.y
            or b.y + b.dy <= a.y
            or a.z + a.dz <= b.z
            or b.z + b.dz <= a.z
        )

    for _ in range(len(packed)):
        moved = False
        for p in packed:
            for attr, dim_attr in [("x", "dx"), ("y", "dy"), ("z", "dz")]:
                old_val = getattr(p, attr)
                setattr(p, attr, 0.0)
                # Find the lowest valid position
                best = 0.0
                for other in packed:
                    if other is p:
                        continue
                    if overlaps(p, other):
                        # Push past this other product
                        candidate = getattr(other, attr) + getattr(other, dim_attr)
                        if candidate > best:
                            best = candidate
                            setattr(p, attr, best)
                if best < old_val:
                    setattr(p, attr, best)
                    moved = True
                else:
                    setattr(p, attr, old_val)
        if not moved:
            break

    return packed
