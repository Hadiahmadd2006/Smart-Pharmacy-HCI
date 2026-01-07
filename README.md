# Smart Pharmacy HCI Project

Interactive pharmacy demo that combines an AR-style product view, a circular menu,
and vision-based login. The UI is built with PySide6 and communicates with a
TUIO/marker source over a local socket.

## Features
- Customer/Admin tabs with a rotating circular menu.
- Product overlays with rotation feedback.
- DeepFace login + emotion tracking.
- Gaze heatmap logging.
- Live camera preview (webcam).

## Requirements
- Windows/macOS/Linux with a webcam.
- Python 3.9+ recommended.
- A local marker/TUIO source that connects to `127.0.0.1:5055` (optional but
  needed for marker-driven interactions).

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

## Run
```bash
python python/Main.py
```

## Marker/TUIO Input
The app opens a socket server on `127.0.0.1:5055` and expects JSON messages with
`event`, `id`, `angle`, `x`, and `y`. You can run `Java/TUIOReceiver.java` or any
client that sends those fields.

## Face Enrollment (SQLite)
Place face images in `python/faces/` using one of these name formats:
- `admin__Alice.jpg`
- `customer__Bob.png`

Data is stored in `python/database/database.sql` (loaded into memory at runtime).

## Gaze Analytics
Heatmap images are saved to `python/heatmaps/` on exit.

## Controls (from the UI guide)
- Card 10: open customer menu
- Card 8: open admin menu
- Card 9: select highlighted item
- Cards 0-3: add products
- Card 11: confirm payment
