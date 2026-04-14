import gurobipy as gp
from gurobipy import GRB

from constants import MAX_FREE_VOL_RATIO
from models import Box, PackedProduct, Product

from .common import (
    _add_bounding_box_vars,
    _add_rotation_constraints,
    _add_separation_constraints,
    _create_position_vars,
)


def milp_stage_1(
    data: list[Product],
    box: Box,
    time_limit: int,
) -> tuple[gp.Model, Box, list[PackedProduct]]:
    n_products = len(data)
    n_range = range(n_products)
    model = gp.Model("3d_packing")
    model.Params.TimeLimit = time_limit
    xs, ys, zs, dx, dy, dz = _create_position_vars(model, data, box, n_range)
    _add_rotation_constraints(model, data, n_range, dx, dy, dz)

    _add_separation_constraints(model, xs, ys, zs, dx, dy, dz, n_products, box)

    max_x, max_y, max_z = _add_bounding_box_vars(
        model, xs, ys, zs, dx, dy, dz, n_range, vtype=GRB.INTEGER
    )

    # Symmetry-breaking
    for i in range(1, n_products):
        model.addConstr((xs[i] + ys[i] + zs[i]) >= (xs[i - 1] + ys[i - 1] + zs[i - 1]))

    # Constrain free volume: box volume <= MAX_FREE_VOL_RATIO * product volume
    total_product_vol = sum(p.width * p.length * p.depth for p in data)
    v_xy = model.addVar(name="v_xy")
    model.addConstr(v_xy == max_x * max_y)
    model.addConstr(v_xy * max_z <= MAX_FREE_VOL_RATIO * total_product_vol)
    model.Params.NonConvex = 2

    model.setObjective(max_x + max_y + max_z, GRB.MINIMIZE)
    model.optimize()

    optimal_box = Box(
        x=max_x.X,
        y=max_y.X,
        z=max_z.X,
    )
    packed = [
        PackedProduct(
            product=data[i],
            x=xs[i].X,
            y=ys[i].X,
            z=zs[i].X,
            dx=dx[i].X,
            dy=dy[i].X,
            dz=dz[i].X,
        )
        for i in n_range
    ]
    return model, optimal_box, packed
