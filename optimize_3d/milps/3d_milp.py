import gurobipy as gp
from gurobipy import GRB

from optimize_3d.constants import (
    W_EXCEED_PENALTY,
    W_FLAT,
    W_FRAGILITY,
    W_GRAVITY,
    W_PYRAMID,
)
from optimize_3d.models import Box, PackedProduct, Product

from ..optimize_3d.milps.common import (
    ROTATION_DIMS,
    _add_bounding_box_vars,
    _add_rotation_constraints,
    _add_separation_constraints,
    _create_position_vars,
)


def _weight_norm(weight: float, min_w: float, max_w: float) -> float:
    return (weight**2 - min_w**2) / (max_w**2 - min_w**2 or 1)


def _face_stats(dims: list[int]) -> tuple[float, float, float]:
    """(min_face, max_face, face_diff) across the 3 unique bottom-face areas."""
    areas = [dims[a] * dims[b] for a, b in [(0, 1), (0, 2), (1, 2)]]
    lo, hi = min(areas), max(areas)
    return lo, hi, hi / lo


def _face_expr(dims: list[int], rotations_i: list[gp.Var]) -> gp.LinExpr:
    """Bottom-face area as a linear expression over rotation binaries."""
    return gp.quicksum(
        rotations_i[r] * dims[ROTATION_DIMS[r][0]] * dims[ROTATION_DIMS[r][1]]
        for r in range(6)
    )


def _build_stage2_objective(
    model: gp.Model,
    data: list[Product],
    n_range: range,
    xs: list[gp.Var],
    ys: list[gp.Var],
    zs: list[gp.Var],
    rotations: list[list[gp.Var]],
    dz: dict[int, gp.Var],
    exceed_x: gp.Var,
    exceed_y: gp.Var,
    exceed_z: gp.Var,
    Mz: float,
) -> gp.QuadExpr:
    """Assemble weighted objective terms for stage-2 packing quality.

    Height is always measured along the Z axis; the caller is responsible
    for remapping axes afterwards if a different face is the bottom.
    """
    min_w = min(d.weight for d in data)
    max_w = max(d.weight for d in data)

    dims_list = [[d.width, d.length, d.depth] for d in data]

    bottom_faces = [_face_expr(dims_list[i], rotations[i]) for i in n_range]

    face_bounds = [_face_stats(dims)[:2] for dims in dims_list]
    abs_min = min(lo for lo, _ in face_bounds)
    abs_max = max(hi for _, hi in face_bounds)

    f_min = model.addVar(
        name="f_min_face", lb=abs_min, ub=abs_max, vtype=GRB.CONTINUOUS
    )
    f_max = model.addVar(
        name="f_max_face", lb=abs_min, ub=abs_max, vtype=GRB.CONTINUOUS
    )
    for i in n_range:
        model.addConstr(f_min <= bottom_faces[i])
        model.addConstr(f_max >= bottom_faces[i])

    face_norm_glob = [
        model.addVar(name=f"fno_{i}", lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS)
        for i in n_range
    ]
    for i in n_range:
        model.addConstr(face_norm_glob[i] * (f_max - f_min) == bottom_faces[i] - f_min)

    obj = gp.QuadExpr()
    for i in n_range:
        dims = dims_list[i]
        w_norm = _weight_norm(data[i].weight, min_w, max_w)
        min_face, max_face, face_diff = _face_stats(dims)

        face_norm = 1 - (bottom_faces[i] - min_face) / (max_face - min_face or 1)

        obj += W_FLAT * face_norm * face_diff  # TODO: why mul with face diff?
        obj += W_PYRAMID * face_norm_glob[i] * zs[i]
        obj += W_GRAVITY * zs[i] * w_norm

    for i in n_range:
        for j in n_range:
            if j == i:
                continue
            # NOTE: using separation constraints here would eliminate n^2
            # "above" variables. Keeping it this way to reduce model complexity.
            above = model.addVar(name=f"{j}_above_{i}", vtype=GRB.BINARY)
            model.addConstr(zs[j] - zs[i] - dz[i] + 0.1 <= Mz * above)
            obj += W_FRAGILITY * (data[j].weight / data[i].weight) * above

    obj += (W_EXCEED_PENALTY * exceed_x) ** 2
    obj += (W_EXCEED_PENALTY * exceed_y) ** 2
    obj += (W_EXCEED_PENALTY * exceed_z) ** 2

    return obj


def milp_stage_2(
    data: list[Product],
    box: Box,
    warm_start: list[PackedProduct],
    time_limit: int,
) -> tuple[gp.Model, list[PackedProduct]]:
    n_products = len(data)
    n_range = range(n_products)
    model = gp.Model("3d_packing_stage2")
    model.Params.TimeLimit = time_limit
    model.Params.NonConvex = 2
    xs, ys, zs, dx, dy, dz = _create_position_vars(
        model,
        data,
        box,
        n_range,
        warm_start=warm_start,
    )
    rotations = _add_rotation_constraints(
        model,
        data,
        n_range,
        dx,
        dy,
        dz,
    )

    _add_separation_constraints(model, xs, ys, zs, dx, dy, dz, n_products, box)

    max_x, max_y, max_z = _add_bounding_box_vars(model, xs, ys, zs, dx, dy, dz, n_range)

    exceed_x = model.addVar(name="exceed_x", lb=0, vtype=GRB.CONTINUOUS)
    exceed_y = model.addVar(name="exceed_y", lb=0, vtype=GRB.CONTINUOUS)
    exceed_z = model.addVar(name="exceed_z", lb=0, vtype=GRB.CONTINUOUS)
    model.addConstr(exceed_x >= max_x - box.x)
    model.addConstr(exceed_y >= max_y - box.y)
    model.addConstr(exceed_z >= max_z - box.z)
    total_exceed = model.addVar(name="total_exceed", lb=0, vtype=GRB.CONTINUOUS)
    model.addConstr(total_exceed == exceed_x + exceed_y + exceed_z)

    obj = _build_stage2_objective(
        model,
        data,
        n_range,
        xs,
        ys,
        zs,
        rotations=rotations,
        dz=dz,
        exceed_x=exceed_x,
        exceed_y=exceed_y,
        exceed_z=exceed_z,
        Mz=box.z,
    )

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    packed = [
        PackedProduct(
            product=data[i],
            x=xs[i].getAttr("x"),
            y=ys[i].getAttr("x"),
            z=zs[i].getAttr("x"),
            dx=dx[i].getAttr("x"),
            dy=dy[i].getAttr("x"),
            dz=dz[i].getAttr("x"),
        )
        for i in n_range
    ]

    return model, packed
