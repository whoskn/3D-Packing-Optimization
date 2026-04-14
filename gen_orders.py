#!/usr/bin/env python3
import argparse
import csv
import json

import numpy as np

from models import Product


def parse_products_csv(filepath: str = "products.csv") -> list[Product]:
    products: list[Product] = [
        None,
    ]
    with open(filepath, "r", newline="") as f:
        for row in csv.DictReader(f):
            length, width, depth, product_id = (
                int(row["Length cm"]),
                int(row["Width cm"]),
                int(row["Depth cm"]),
                int(row["product_id"]),
            )
            products.append(
                Product(
                    avg_sales=int(row["Average Sales per Week"]),
                    weight=float(row["weight KG"]),
                    length=length,
                    width=width,
                    depth=depth,
                    product_id=product_id
                )
            )
    return products


def generate_orders(
    data: list[Product],
    param: float,
    min_packages: int,
    max_packages: int,
    size_index: float,
    weight_index: float,
    max_orders: int | None = None,
) -> list[list[int]]:
    orders = []

    n_products = len(data)
    available = np.ones(n_products, dtype=bool)
    volumes = np.array([p.volume for p in data])
    weights = np.array([p.weight for p in data])

    while np.any(available):
        if max_orders is not None and len(orders) >= max_orders:
            break

        available_indices = np.where(available)[0]
        n_available = len(available_indices)

        order_size = np.clip(np.random.geometric(param), min_packages, max_packages)
        order_size = min(order_size, n_available)

        if order_size < min_packages and n_available >= min_packages:
            order_size = min_packages

        if n_available < min_packages:
            if orders:
                orders[-1].extend(available_indices.tolist())
            else:
                orders.append(available_indices.tolist())
            break

        start_idx = np.random.choice(available_indices)

        # Build order iteratively to ensure ALL products are mutually compatible
        order = [start_idx]
        order_min_vol = volumes[start_idx]
        order_max_vol = volumes[start_idx]
        order_min_weight = weights[start_idx]
        order_max_weight = weights[start_idx]

        remaining_mask = available.copy()
        remaining_mask[start_idx] = False

        while len(order) < order_size and np.any(remaining_mask):
            remaining_indices = np.where(remaining_mask)[0]
            rem_volumes = volumes[remaining_indices]
            rem_weights = weights[remaining_indices]

            # Check compatibility with current order bounds
            new_max_vols = np.maximum(order_max_vol, rem_volumes)
            new_min_vols = np.minimum(order_min_vol, rem_volumes)
            new_max_weights = np.maximum(order_max_weight, rem_weights)
            new_min_weights = np.minimum(order_min_weight, rem_weights)

            vol_compatible = (
                (new_max_vols * size_index <= new_min_vols)
                if size_index > 0
                else np.ones(len(remaining_indices), dtype=bool)
            )
            weight_compatible = (
                (new_max_weights * weight_index <= new_min_weights)
                if weight_index > 0
                else np.ones(len(remaining_indices), dtype=bool)
            )

            compatible_mask = vol_compatible & weight_compatible
            compatible_indices = remaining_indices[compatible_mask]

            if len(compatible_indices) == 0:
                break

            next_idx = np.random.choice(compatible_indices)
            order.append(next_idx)
            remaining_mask[next_idx] = False

            # Update bounds
            order_min_vol = min(order_min_vol, volumes[next_idx])
            order_max_vol = max(order_max_vol, volumes[next_idx])
            order_min_weight = min(order_min_weight, weights[next_idx])
            order_max_weight = max(order_max_weight, weights[next_idx])

        orders.append(order)
        available[order] = False

    return orders


if __name__ == "__main__":
    from constants import (
        GEN_ORDERS,
        GEN_MAX_PACKAGES,
        GEN_MIN_PACKAGES,
        GEN_ORDER_PARAM,
        GEN_SIZE_INDEX,
        GEN_WEIGHT_INDEX,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Output file to save generated orders (JSON)")
    args = parser.parse_args()

    products = parse_products_csv()
    orders = generate_orders(
        data=products,
        param=GEN_ORDER_PARAM,
        min_packages=GEN_MIN_PACKAGES,
        max_packages=GEN_MAX_PACKAGES,
        size_index=GEN_SIZE_INDEX,
        weight_index=GEN_WEIGHT_INDEX,
        max_orders=GEN_ORDERS,
    )

    with open(args.filename, "w") as f:
        json.dump([[int(i) for i in order] for order in orders], f)
    print(f"Saved {len(orders)} orders to {args.filename}")
