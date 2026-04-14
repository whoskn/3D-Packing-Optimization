#!/usr/bin/env python3
import argparse
import json

from gen_orders import parse_products_csv
from models import Box
from pipelines import pipeline, pipeline_fixed_boxes
from visualize_3d import visualize_packing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename", help="JSON file with orders (output of gen_orders.py)"
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

    data = parse_products_csv()

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
