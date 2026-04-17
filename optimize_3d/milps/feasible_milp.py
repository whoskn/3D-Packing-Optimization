import gurobipy as gp
from gurobipy import GRB

from optimize_3d.models import Box, Product

from .common import (
    _add_rotation_constraints,
    _add_separation_constraints,
    _create_position_vars,
)


def milp_feasibility(
    data: list[Product],
    box: Box,
    time_limit: int,
) -> bool:
    """Check whether *data* can be packed into *box*. Returns True if feasible."""
    n_products = len(data)
    n_range = range(n_products)
    model = gp.Model("3d_feasibility")
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 0
    xs, ys, zs, dx, dy, dz = _create_position_vars(model, data, box, n_range)
    _add_rotation_constraints(model, data, n_range, dx, dy, dz)
    _add_separation_constraints(model, xs, ys, zs, dx, dy, dz, n_products, box)

    for i in n_range:
        model.addConstr(xs[i] + dx[i] <= box.x)
        model.addConstr(ys[i] + dy[i] <= box.y)
        model.addConstr(zs[i] + dz[i] <= box.z)

    model.setObjective(0, GRB.MINIMIZE)
    model.optimize()

    return model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL)
