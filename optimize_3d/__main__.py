#!/usr/bin/env python3
import argparse
import json

from optimize_3d.gen_orders import parse_products_csv
from optimize_3d.models import Box
from optimize_3d.pipelines import pipeline, pipeline_fixed_boxes
from optimize_3d.visualize_3d import visualize_packing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", help="JSON file with orders (output of gen_orders.py)"
    )
    parser.add_argument(
        "--products",
        "-p",
        help="CSV file with product definitions (default: products.csv)",
        default="products.csv",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--start",
        "-s",
        help="From which order in the `filename` to start with",
        default=0,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--cartons",
        "-c",
        help="JSON file with available carton (box) definitions",
        required=False,
        type=str,
    )
    args = parser.parse_args()

    data = parse_products_csv(args.products)

    boxes = None
    if args.cartons:
        with open(args.cartons) as f:
            raw_boxes = json.load(f)
        boxes = [Box(x=b["length"], y=b["width"], z=b["height"]) for b in raw_boxes]

    with open(args.filename) as f:
        orders = json.load(f)

    for o in range(args.start, len(orders)):
        order = orders[o]
        packages = [data[p] for p in order]
        if boxes:
            results = pipeline_fixed_boxes(packages, boxes)
        else:
            results = pipeline(packages)
        visualize_packing(results)


if __name__ == "__main__":
    main()
