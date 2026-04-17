import gurobipy as gp
from gurobipy import GRB

from optimize_3d.constants import (
    SEP_HEAVY_FLOOR,
    SEP_MAX_PER_BIN,
    SEP_WEIGHT_RATIO,
    SEP_WEIGHT_THRESHOLD,
)
from optimize_3d.models import Product


def _milp_bin_pack(products: list[Product]) -> list[list[Product]]:
    """MILP bin-packing: minimize number of bins subject to weight,
    size, and compatibility constraints."""
    n = len(products)
    weights = [p.weight for p in products]

    max_bins = n

    model = gp.Model("separate_order")
    # model.Params.OutputFlag = 0

    x = model.addVars(n, max_bins, vtype=GRB.BINARY, name="x")
    y = model.addVars(max_bins, vtype=GRB.BINARY, name="y")

    for i in range(n):
        model.addConstr(gp.quicksum(x[i, b] for b in range(max_bins)) == 1)

    for b in range(max_bins):
        model.addConstr(
            gp.quicksum(weights[i] * x[i, b] for i in range(n))
            <= SEP_WEIGHT_THRESHOLD * y[b]
        )

    # Max products per bin
    for b in range(max_bins):
        model.addConstr(gp.quicksum(x[i, b] for i in range(n)) <= SEP_MAX_PER_BIN)

    # Incompatible pairs: weight ratio >= SEP_WEIGHT_RATIO and heavier > SEP_HEAVY_FLOOR
    for i in range(n):
        for j in range(i + 1, n):
            hi, lo = max(weights[i], weights[j]), min(weights[i], weights[j])
            if hi >= SEP_HEAVY_FLOOR and (lo == 0 or hi / lo >= SEP_WEIGHT_RATIO):
                for b in range(max_bins):
                    model.addConstr(x[i, b] + x[j, b] <= 1)

    for b in range(1, max_bins):
        model.addConstr(y[b] <= y[b - 1])

    model.setObjective(gp.quicksum(y[b] for b in range(max_bins)), GRB.MINIMIZE)
    model.optimize()

    sub_orders: list[list[Product]] = []
    for b in range(max_bins):
        if y[b].getAttr("x") < 0.5:
            continue
        group = [products[i] for i in range(n) if x[i, b].getAttr("x") > 0.5]
        sub_orders.append(group)

    return sub_orders


def separate_order(order: list[Product]) -> list[list[Product]]:
    """Pre-process an order and split it into sub-orders via MILP bin-packing."""
    # Pull out products that alone exceed the threshold — each becomes its own order
    heavy = [p for p in order if p.weight > SEP_WEIGHT_THRESHOLD]
    remaining = [p for p in order if p.weight <= SEP_WEIGHT_THRESHOLD]

    # TODO: check the logic
    solo_orders: list[list[Product]] = [[p] for p in heavy]

    if not remaining:
        return solo_orders

    total_weight = sum(p.weight for p in remaining)
    if total_weight <= SEP_WEIGHT_THRESHOLD and len(remaining) <= SEP_MAX_PER_BIN:
        return [remaining] + solo_orders

    return _milp_bin_pack(remaining) + solo_orders
