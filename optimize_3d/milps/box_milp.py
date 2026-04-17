import gurobipy as gp
from gurobipy import GRB

from optimize_3d.constants import MAX_FREE_VOL, SEP_MAX_PER_BIN
from optimize_3d.models import Box, PackedProduct, Product

from .common import _add_rotation_constraints


def milp_box_selection(
    data: list[Product],
    boxes: list[Box],
    time_limit: int,
) -> list[tuple[list[PackedProduct], Box]]:
    """Assign products to boxes minimizing bin count, with full 3D feasibility.

    Each box type can be used unlimited times. Returns a list of
    (packed_products, selected_box) tuples, one per used bin.
    """
    N = len(data)
    K = len(boxes)
    B = N  # max bins = number of products

    Mx = max(b.x for b in boxes)
    My = max(b.y for b in boxes)
    Mz = max(b.z for b in boxes)
    vol = {k: boxes[k].x * boxes[k].y * boxes[k].z for k in range(K)}

    model = gp.Model("box_selection")
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 1

    a = {
        (i, b): model.addVar(name=f"a_{i}_{b}", vtype=GRB.BINARY)
        for i in range(N)
        for b in range(B)
    }
    y = [model.addVar(name=f"y_{b}", vtype=GRB.BINARY) for b in range(B)]
    t = {
        (b, k): model.addVar(name=f"t_{b}_{k}", vtype=GRB.BINARY)
        for b in range(B)
        for k in range(K)
    }

    # Position UBs use max box dims; fits-in-box constraints enforce per-bin limits
    xs, ys, zs = [], [], []
    dxs, dys, dzs = [], [], []
    for i in range(N):
        min_dim = min(data[i].width, data[i].length, data[i].depth)
        xs.append(model.addVar(name=f"xs_{i}", lb=0, ub=Mx - min_dim))
        ys.append(model.addVar(name=f"ys_{i}", lb=0, ub=My - min_dim))
        zs.append(model.addVar(name=f"zs_{i}", lb=0, ub=Mz - min_dim))
        dxs.append(model.addVar(name=f"dx_{i}", vtype=GRB.INTEGER))
        dys.append(model.addVar(name=f"dy_{i}", vtype=GRB.INTEGER))
        dzs.append(model.addVar(name=f"dz_{i}", vtype=GRB.INTEGER))

    _add_rotation_constraints(model, data, range(N), dxs, dys, dzs)

    same_bin = {
        (i, j): model.addVar(name=f"same_{i}_{j}", vtype=GRB.BINARY)
        for i in range(N)
        for j in range(i + 1, N)
    }
    sep = {
        (i, j, d): model.addVar(name=f"sep_{i}_{j}_{d}", vtype=GRB.BINARY)
        for i in range(N)
        for j in range(i + 1, N)
        for d in range(6)
    }

    # Each product in at most one bin; only active bins accept products
    for i in range(N):
        model.addConstr(gp.quicksum(a[i, b] for b in range(B)) <= 1, name=f"assign_{i}")
        for b in range(B):
            model.addConstr(a[i, b] <= y[b], name=f"activate_{i}_{b}")

    # Active bin gets exactly one box type
    for b in range(B):
        model.addConstr(
            gp.quicksum(t[b, k] for k in range(K)) == y[b], name=f"type_{b}"
        )

    # Products must fit within their assigned box
    axes = [(xs, dxs, Mx, "x"), (ys, dys, My, "y"), (zs, dzs, Mz, "z")]
    for i in range(N):
        for b in range(B):
            for pos, dim, M, attr in axes:
                box_dim = gp.quicksum(
                    t[b, k] * getattr(boxes[k], attr) for k in range(K)
                )
                model.addConstr(
                    pos[i] + dim[i] <= box_dim + M * (1 - a[i, b]),
                    name=f"fit{attr}_{i}_{b}",
                )

    # same_bin linking
    for i in range(N):
        for j in range(i + 1, N):
            for b in range(B):
                model.addConstr(
                    same_bin[i, j] >= a[i, b] + a[j, b] - 1,
                    name=f"samebin_{i}_{j}_{b}",
                )

    # Non-overlap: big-M separation on at least one axis when sharing a bin
    for i in range(N):
        for j in range(i + 1, N):
            sb = same_bin[i, j]
            d = 0
            for pos, dim, M, _ in axes:
                model.addConstr(
                    pos[i] + dim[i] <= pos[j] + M * (1 - sep[i, j, d]) + M * (1 - sb),
                    name=f"novlp_{d}_{i}_{j}",
                )
                model.addConstr(
                    pos[j] + dim[j]
                    <= pos[i] + M * (1 - sep[i, j, d + 1]) + M * (1 - sb),
                    name=f"novlp_{d + 1}_{i}_{j}",
                )
                d += 2
            model.addConstr(
                gp.quicksum(sep[i, j, d] for d in range(6)) >= sb,
                name=f"sep_sum_{i}_{j}",
            )

    # Max products per bin
    for b in range(B):
        model.addConstr(
            gp.quicksum(a[i, b] for i in range(N)) <= SEP_MAX_PER_BIN,
            name=f"max_per_bin_{b}",
        )

    # Symmetry breaking: bins used in order
    for b in range(B - 1):
        model.addConstr(y[b] >= y[b + 1], name=f"sym_{b}")

    # Products must fill at least (1 - MAX_FREE_VOL) of their box
    for b in range(B):
        product_vol = gp.quicksum(a[i, b] * data[i].volume for i in range(N))
        box_vol = gp.quicksum(t[b, k] * vol[k] for k in range(K))
        model.addConstr(
            product_vol >= (1 - MAX_FREE_VOL) * box_vol, name=f"free_vol_{b}"
        )

    UNPACK_PENALTY = 100
    unpacked = [1 - gp.quicksum(a[i, b] for b in range(B)) for i in range(N)]
    model.setObjective(
        gp.quicksum(y[b] for b in range(B)) + UNPACK_PENALTY * gp.quicksum(unpacked),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(
            f"Box selection MILP infeasible or timed out (status={model.Status})"
        )

    results: list[tuple[list[PackedProduct], Box]] = []
    for b in range(B):
        if y[b].X < 0.5:
            continue
        selected_k = max(range(K), key=lambda k: t[b, k].X)
        packed = [
            PackedProduct(
                product=data[i],
                x=xs[i].X,
                y=ys[i].X,
                z=zs[i].X,
                dx=round(dxs[i].X),
                dy=round(dys[i].X),
                dz=round(dzs[i].X),
            )
            for i in range(N)
            if a[i, b].X > 0.5
        ]
        results.append((packed, boxes[selected_k]))

    unpacked_products = [
        data[i] for i in range(N) if sum(a[i, b].X for b in range(B)) < 0.5
    ]
    return results, unpacked_products
