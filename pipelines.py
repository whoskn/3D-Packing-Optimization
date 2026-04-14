from constants import GUR_TIME_LIMIT, BOX_LIMIT
from milps import milp_box_selection, milp_stage_1, milp_stage_2, separate_order
from models import Box, PackedProduct, Product


def pipeline(
    packages: list[Product],
) -> list[tuple[list[PackedProduct], Box, float]]:
    """Run order separation then both MILP stages on each sub-order.

    Returns a list of (packed, box, solver_obj) tuples — one per carton.
    """
    sub_orders = separate_order(packages)
    results = []

    for sub in sub_orders:
        sub.sort(key=lambda p: p.weight, reverse=True)

        if len(sub) == 1:
            p = sub[0]
            packed = [PackedProduct(product=p, dx=p.width, dy=p.length, dz=p.depth)]
            box = Box(x=p.width, y=p.length, z=p.depth)
            remapped_box = box.remap_axis(packed)
            results.append((packed, remapped_box, 0.0))
            continue

        _, optimal_box, packed = milp_stage_1(sub, BOX_LIMIT, time_limit=GUR_TIME_LIMIT)

        remapped_box = optimal_box.remap_axis(packed)

        model, packed = milp_stage_2(
            sub,
            remapped_box,
            packed,
            time_limit=GUR_TIME_LIMIT,
        )
        solver_obj = model.ObjVal

        # packed = pull_to_origin(packed)
        results.append((packed, remapped_box, solver_obj))

    return results


def pipeline_fixed_boxes(
    packages: list[Product],
    boxes: list[Box],
) -> list[tuple[list[PackedProduct], Box | None, float]]:
    """Pack products into boxes using a box-selection MILP.

    For each sub-order produced by order separation, a single MILP finds the
    optimal assignment of products to box types (minimizing bin count, then
    volume). Each bin's solution is then refined by stage 2 for quality.

    Returns a list of (packed, box, solver_obj) tuples — one per carton.
    """
    sub_orders = separate_order(packages)

    # for o in sub_orders:
    #     print([p.product_id for p in o])
    #     print()

    # exit(0)

    # [[69189, 77250, 54634, 89139, 90985],
    # [68651, 67190, 83447, 58466, 6571, 58058],
    # [56966, 30067, 16212, 34699, 87622, 12334],
    # [59110],
    # [9059],
    # [6222],]

    # sub_orders = [packages]
    results: list[tuple[list[PackedProduct], Box | None, float]] = []

    for sub in sub_orders:
        sub.sort(key=lambda p: p.weight, reverse=True)

        # Filter out products that don't fit in any available box
        packable, unpackable = [], []
        boxes_sorted_dims = [sorted([b.x, b.y, b.z]) for b in boxes]
        for p in sub:
            dims = sorted([p.width, p.length, p.depth])
            if any(
                dims[0] <= bd[0] and dims[1] <= bd[1] and dims[2] <= bd[2]
                for bd in boxes_sorted_dims
            ):
                packable.append(p)
            else:
                unpackable.append(p)

        for p in unpackable:
            packed = [PackedProduct(product=p, dx=p.width, dy=p.length, dz=p.depth)]
            results.append((packed, None, 0.0))

        if not packable:
            continue

        bins, unpacked = milp_box_selection(packable, boxes, time_limit=GUR_TIME_LIMIT)

        for p in unpacked:
            packed = [PackedProduct(product=p, dx=p.width, dy=p.length, dz=p.depth)]
            results.append((packed, None, 0.0))

        for bin_packed, bin_box in bins:
            if len(bin_packed) == 1:
                remapped_box = bin_box.remap_axis(bin_packed)
                results.append((bin_packed, remapped_box, 0.0))
                continue

            bin_products = [pp.product for pp in bin_packed]
            remapped_box = bin_box.remap_axis(bin_packed)

            # results.append((bin_packed, remapped_box, 0.0))

            model, packed = milp_stage_2(
                bin_products,
                remapped_box,
                warm_start=bin_packed,
                time_limit=GUR_TIME_LIMIT,
            )
            results.append((packed, remapped_box, model.ObjVal))

    return results
