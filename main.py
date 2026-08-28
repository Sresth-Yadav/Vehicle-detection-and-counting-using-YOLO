"""
main.py
-------
Vehicle Tracking & Counting — CLI entry point.

Runs YOLOv8 detection + ByteTrack tracking over a video, counts vehicles
that cross a configurable line (broken down by class and direction), and
writes an annotated output video plus a CSV log of every counting event.

Usage:
    python main.py --source data/traffic.mp4 --output outputs/result.mp4
    python main.py --source 0                      # webcam
    python main.py --source data/traffic.mp4 --show # live preview window
"""

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
import yaml

from src.detector import VehicleDetector
from src.counter import LineCounter
from src.visualization import draw_tracks, draw_trails, draw_line, draw_counts_panel


def parse_args():
    parser = argparse.ArgumentParser(description="Vehicle Tracking & Counting")
    parser.add_argument("--source", type=str, required=True,
                         help="Path to input video file, or '0' for webcam")
    parser.add_argument("--output", type=str, default="outputs/result.mp4",
                         help="Path to save annotated output video")
    parser.add_argument("--config", type=str, default="config.yaml",
                         help="Path to config YAML")
    parser.add_argument("--log", type=str, default="outputs/count_log.csv",
                         help="Path to save CSV log of counting events")
    parser.add_argument("--show", action="store_true",
                         help="Display a live preview window while processing")
    parser.add_argument("--no-save", action="store_true",
                         help="Skip writing the annotated output video (faster)")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def open_source(source: str):
    src = 0 if source == "0" else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")
    return cap


def main():
    args = parse_args()
    cfg = load_config(args.config)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)

    detector = VehicleDetector(
        model_path=cfg["model_path"],
        vehicle_classes=cfg["vehicle_classes"],
        confidence=cfg["confidence"],
        iou=cfg["iou"],
        tracker_cfg=cfg["tracker"],
    )

    counter = LineCounter(
        point1=cfg["counting_line"]["point1"],
        point2=cfg["counting_line"]["point2"],
        direction_labels=cfg["direction_labels"],
        trail_length=cfg["draw"]["trail_length"],
    )

    cap = open_source(args.source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    start_time = time.time()

    print(f"[INFO] Processing '{args.source}' ({width}x{height} @ {fps:.1f}fps)")
    print(f"[INFO] Counting line: {cfg['counting_line']['point1']} -> {cfg['counting_line']['point2']}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            tracks = detector.track(frame)

            for t in tracks:
                counter.update(t.track_id, t.class_name, t.centroid, frame_idx)

            draw_conf = cfg["draw"]
            frame = draw_line(frame, cfg["counting_line"]["point1"],
                               cfg["counting_line"]["point2"], draw_conf["line_thickness"])
            if draw_conf["show_trail"]:
                frame = draw_trails(frame, counter.trails)
            frame = draw_tracks(frame, tracks, draw_conf["box_thickness"], draw_conf["font_scale"])
            frame = draw_counts_panel(frame, counter.breakdown())

            if writer is not None:
                writer.write(frame)

            if args.show:
                cv2.imshow("Vehicle Tracking & Counting", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] Interrupted by user.")
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f"[INFO] Frame {frame_idx} | elapsed {elapsed:.1f}s | "
                      f"running total: {counter.total()}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print("\n===== SUMMARY =====")
    print(f"Frames processed : {frame_idx}")
    print(f"Time elapsed     : {elapsed:.1f}s ({frame_idx / max(elapsed, 1e-6):.1f} fps avg)")
    for direction, classes in counter.breakdown().items():
        total = sum(classes.values())
        print(f"{direction}: {total}  {dict(classes)}")
    print(f"Grand total      : {counter.total()}")

    if counter.event_log:
        pd.DataFrame(counter.event_log).to_csv(args.log, index=False)
        print(f"[INFO] Event log saved to {args.log}")
    if writer is not None:
        print(f"[INFO] Annotated video saved to {args.output}")


if __name__ == "__main__":
    main()
