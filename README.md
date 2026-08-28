# 🚗 Vehicle Tracking & Counting

Real-time vehicle detection, multi-object tracking, and line-crossing
counting from video, built with **YOLOv8** and **ByteTrack**.

Detects cars, trucks, buses, and motorcycles in a video stream, assigns
each one a persistent ID as it moves across frames, and counts how many
cross a user-defined line — broken down by vehicle class and direction
(IN / OUT).

## Features

- **Detection** — YOLOv8 (Ultralytics) object detection, restricted to
  vehicle classes (car, truck, bus, motorcycle).
- **Tracking** — ByteTrack multi-object tracking gives each vehicle a
  stable ID across frames, so it's only counted once.
- **Counting** — configurable virtual line; a vehicle is counted the
  moment its centroid crosses it, tagged with direction (IN/OUT) and
  class.
- **Visualization** — bounding boxes, track IDs, motion trails, the
  counting line, and a live counts panel drawn on every frame.
- **Logging** — every counting event (track ID, class, direction, frame
  number) is saved to a CSV for later analysis.
- **Config-driven** — tweak the model, thresholds, and counting line via
  `config.yaml` without touching code.
- Works on video files or a live webcam feed.

## Demo

```
$ python main.py --source data/traffic.mp4 --output outputs/result.mp4

[INFO] Processing 'data/traffic.mp4' (1280x720 @ 30.0fps)
[INFO] Counting line: [0, 432] -> [1280, 432]
[INFO] Frame 100 | elapsed 4.2s | running total: 6
[INFO] Frame 200 | elapsed 8.1s | running total: 13
...
===== SUMMARY =====
Frames processed : 450
Time elapsed     : 18.6s (24.2 fps avg)
IN: 21  {'car': 16, 'truck': 3, 'bus': 1, 'motorcycle': 1}
OUT: 17  {'car': 14, 'truck': 2, 'bus': 1}
Grand total      : 38
[INFO] Event log saved to outputs/count_log.csv
[INFO] Annotated video saved to outputs/result.mp4
```

## Project Structure

```
vehicle-tracking-counting/
├── main.py                # CLI entry point — runs the full pipeline
├── config.yaml             # model, thresholds, counting line settings
├── requirements.txt
├── src/
│   ├── detector.py         # YOLOv8 + ByteTrack wrapper
│   ├── counter.py          # line-crossing counting logic
│   └── visualization.py    # drawing helpers (boxes, trails, panel)
├── data/                   # put input videos here (gitignored)
├── outputs/                # annotated videos + CSV logs land here
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/vehicle-tracking-counting.git
cd vehicle-tracking-counting

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

> The first run automatically downloads `yolov8n.pt` (~6 MB) from
> Ultralytics — no manual model download needed.

## Usage

**Process a video file:**
```bash
python main.py --source data/traffic.mp4 --output outputs/result.mp4
```

**Use a webcam:**
```bash
python main.py --source 0 --show
```

**Preview live while processing (press `q` to stop):**
```bash
python main.py --source data/traffic.mp4 --show
```

**Skip saving the output video (faster, counts only):**
```bash
python main.py --source data/traffic.mp4 --no-save
```

### CLI options

| Flag         | Default                  | Description                                |
|--------------|---------------------------|---------------------------------------------|
| `--source`   | *(required)*              | Video path, or `0` for webcam                |
| `--output`   | `outputs/result.mp4`      | Annotated output video path                  |
| `--config`   | `config.yaml`             | Path to config file                          |
| `--log`      | `outputs/count_log.csv`   | CSV log of counting events                   |
| `--show`     | off                       | Show a live preview window                   |
| `--no-save`  | off                       | Don't write the annotated video              |

## Configuring the counting line

Vehicles are counted when their bounding-box centroid crosses a line you
define in `config.yaml`:

```yaml
counting_line:
  point1: [0, 432]
  point2: [1280, 432]
```

Set these two `(x, y)` pixel points to match a good chokepoint in **your**
video's resolution — e.g. across a road lane, a bridge, or a doorway. A
horizontal line works for traffic moving vertically through frame; use a
vertical or diagonal line for other orientations.

To find good coordinates quickly, open one frame of your video in any
image viewer/editor and read off pixel positions, or run:
```bash
python -c "import cv2; cap=cv2.VideoCapture('data/traffic.mp4'); print(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))"
```

## How it works

1. **Detect** — each frame is passed through YOLOv8, filtered to only
   the COCO classes for car (2), motorcycle (3), bus (5), and truck (7).
2. **Track** — Ultralytics' built-in ByteTrack associates detections
   across frames into persistent tracks with stable IDs.
3. **Count** — for each track, we compute which side of the counting
   line its centroid is on using the sign of a 2D cross product. When
   the sign flips between consecutive frames, that's a line crossing —
   we tally it once per track ID, tagged with class and direction.
4. **Render** — boxes, IDs, trails, the line, and a live count panel are
   drawn onto each frame and written to the output video.

## Customization ideas

- **Higher accuracy**: swap `model_path` in `config.yaml` for
  `yolov8s.pt` / `yolov8m.pt`, or a model fine-tuned on your own traffic
  footage.
- **Multiple counting lines**: instantiate several `LineCounter`
  objects (e.g. one per lane) and feed each track's centroid to all of
  them.
- **Speed estimation**: use the trail history in `LineCounter.trails`
  plus known real-world distance between two lines to estimate vehicle
  speed.
- **Zone-based counting** (instead of a line): replace the cross-product
  side test with a point-in-polygon check.
- **Dashboard**: the CSV log in `outputs/count_log.csv` is ready to feed
  into a pandas notebook or a simple Streamlit/Plotly dashboard for
  hourly traffic charts.

## Requirements

- Python 3.9+
- See `requirements.txt` (Ultralytics YOLOv8, OpenCV, NumPy, Pandas, PyYAML)
- A GPU is not required but significantly speeds up processing on longer
  videos.

## License

MIT — see [LICENSE](LICENSE).
