import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Optional, Tuple
from collections import deque

LOG_VERBOSITY = os.environ.get("SPH_LOG_MODE", "app").strip().lower()
if LOG_VERBOSITY not in {"app", "errors"}:
    LOG_VERBOSITY = "app"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=FutureWarning)

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

import face_rec
from db import init_database, export_database_sql, db_session
from GUI import APP_STYLE, BG, BORDER, CARD, CARD_SOFT, ACCENT, MUTED, TEXT


previous_angles = {}
last_change_time = {}
last_select_time = {}
cart = {}
selected_customer_index = 0
selected_admin_index = 0
selected_manage_index = 0
manage_users_mode = "users"
manage_target_user = None
customer_menu_open = False
admin_menu_open = False
manage_users_open = False
edit_stock_open = False
purchases_open = False
inventory_open = False
customer_history_active = False
calibration_active = False
calibration_targets = []
calibration_index = 0
gaze_point_norm = None
gaze_point_timestamp = 0.0
gaze_point_mapped = None
gaze_point_mapped_timestamp = 0.0
gaze_affine = None
gaze_poly = None
gaze_samples = []
gaze_calibrated = False
gaze_features = None
gaze_features_timestamp = 0.0
gaze_feature_map = None
gaze_stable = False
gaze_stable_timestamp = 0.0
calibration_buffer = []
calibration_points = []
calibration_target_started = None
calibration_capture_until = 0.0
calibration_capture_samples = []
calibration_auto = True
calibration_auto_index = 0
calibration_auto_next = 0.0
customer_menu_locked = False
admin_menu_locked = False
customer_menu_last_index = 0
admin_menu_last_index = 0
edit_stock_mode = "items"
edit_stock_target = None
selected_edit_index = 0
edit_stock_quantity = 0
selected_purchase_index = 0
purchases_mode = "customers"
purchases_target_customer = None
selected_inventory_index = 0
inventory_mode = "items"
inventory_target_item = None
product_map = {0: "Panadol Extra", 1: "Vicks VapoRub", 2: "Epipen", 3: "Strepsils"}
MARKER_ALIAS_DEFAULT = {99: 10}
ar_objects = {}

customer_menu_options = ["Pay Now", "Purchase History", "Change User", "Exit"]
admin_menu_options = [
    "View Inventory",
    "Edit Stock",
    "View Purchases",
    "Reports",
    "Manage Users",
    "Enroll Customer",
    "Change User",
    "Exit",
]

latest_frame = None
frame_lock = threading.Lock()
known_users = []
known_users_lock = threading.Lock()
conflicted_names = set()
face_status_lock = threading.Lock()
current_identity = {"name": "Recognizing user", "role": "unknown", "confidence": 0.0}
current_emotion = "Unknown"
gaze_log = []
gaze_scroll_value = None
gaze_scroll_timestamp = 0.0
heatmap_acc = None
session_heatmap_acc = None
session_active = False
active_customer_name = None
session_lock = threading.Lock()
heatmap_target_size = None
stop_event = threading.Event()
reports_open = False
reports_mode = "customers"
reports_target_customer = None
current_ui_state = "idle"
socket_clients = 0
socket_clients_lock = threading.Lock()
last_action_message = "--"
last_action_lock = threading.Lock()
marker_paths = {}
last_traj_label = {}
last_traj_time = {}

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "database"
FACE_IMAGE_DIR = BASE_DIR / "faces"
IMAGE_DIR = BASE_DIR / "images"
HEATMAP_DIR = BASE_DIR / "heatmaps"
HEATMAP_BG_PATH = HEATMAP_DIR / "customer_screen.png"
HEATMAP_DATA_DIR = HEATMAP_DIR / "_data"
HEATMAP_CAMERA_MASK = (0.66, 0.06, 0.31, 0.30)
GAZE_FLIP_X = True
GAZE_FLIP_Y = True
DATABASE_SQL_PATH = DATA_DIR / "database.sql"
MARKER_MAP_PATH = BASE_DIR.parent / "Java" / "marker_map.json"
FACE_MODEL = "Facenet512"
FACE_MATCH_THRESHOLD = 0.35
FACE_MISSING_TIMEOUT = 180.0
LOW_STOCK_THRESHOLD = 2
GAZE_AUTO_MODE = False
GAZE_SHOW_DOT = False
CURRENCY_LABEL = "EGP"
TRAJ_WINDOW_SEC = 1.6
TRAJ_MIN_POINTS = 6
TRAJ_MIN_DISTANCE = 0.08
TRAJ_STRAIGHTNESS = 0.85
TRAJ_LOOP_CLOSE = 0.05
TRAJ_TURN_THRESHOLD = 0.5
GAZE_SCROLL_DWELL = 0.7
GAZE_SCROLL_INTERVAL = 0.12
GAZE_SCROLL_STEP = 8
GAZE_SCROLL_TOP = 0.35
GAZE_SCROLL_BOTTOM = 0.65
MARKER_SCROLL_STEP = 6
HEATMAP_DOWNSCALE = 1.0
HEATMAP_BLUR = 15
HEATMAP_PERCENTILE = 97
HEATMAP_GAMMA = 1.1
CALIB_DWELL_SEC = 1.0
CALIB_MIN_SAMPLES = 8
CALIB_STABLE_TIMEOUT = 0.6
CALIB_CAPTURE_SEC = 1.2


def log_event(message: str, level: str = "info") -> None:
    global last_action_message
    with last_action_lock:
        last_action_message = message
    if LOG_VERBOSITY == "errors" and level != "error":
        return
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def normalize_marker_id(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        try:
            return int(float(value.strip()))
        except ValueError:
            return None
    return None


def load_marker_aliases() -> dict[int, int]:
    aliases = dict(MARKER_ALIAS_DEFAULT)
    try:
        if MARKER_MAP_PATH.exists():
            data = json.loads(MARKER_MAP_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for raw, mapped in data.items():
                    raw_id = normalize_marker_id(raw)
                    mapped_id = normalize_marker_id(mapped)
                    if raw_id is None or mapped_id is None:
                        continue
                    aliases[raw_id] = mapped_id
    except Exception as exc:
        log_event(f"Failed to load marker_map.json: {exc}", level="error")
    return aliases


marker_alias = load_marker_aliases()


def init_face_db() -> None:
    init_database(DATABASE_SQL_PATH)
    export_database_sql(DATABASE_SQL_PATH)


def seed_inventory_if_empty() -> None:
    should_export = False
    with db_session() as conn:
        row = conn.execute("SELECT COUNT(*) FROM inventory").fetchone()
        if row and row[0] > 0:
            return
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for marker_id, name in product_map.items():
            conn.execute(
                """
                INSERT INTO inventory (name, stock, price, marker_id, image_path, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, 10, 0.0, marker_id, "", now),
            )
        should_export = True
    if should_export:
        export_database_sql(DATABASE_SQL_PATH)


def load_inventory_items() -> list[dict]:
    items = []
    with db_session() as conn:
        cursor = conn.execute(
            """
            SELECT name, stock, price, marker_id, image_path
            FROM inventory
            ORDER BY name
            """
        )
        for name, stock, price, marker_id, image_path in cursor.fetchall():
            items.append(
                {
                    "name": name,
                    "stock": int(stock),
                    "price": float(price),
                    "marker_id": marker_id,
                    "image_path": image_path or "",
                }
            )
    return items


def get_inventory_item_by_marker(pid: int) -> Optional[dict]:
    with db_session() as conn:
        row = conn.execute(
            "SELECT name, stock, price FROM inventory WHERE marker_id = ?",
            (pid,),
        ).fetchone()
    if not row:
        return None
    return {"name": row[0], "stock": int(row[1]), "price": float(row[2])}


def get_inventory_stock_by_marker(pid: int) -> Optional[Tuple[str, int]]:
    item = get_inventory_item_by_marker(pid)
    if not item:
        return None
    return item["name"], item["stock"]


def safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in " _-" else "" for ch in name).strip()
    if not cleaned:
        return "unknown"
    return cleaned.replace(" ", "_").lower()


def format_price(value: float) -> str:
    return f"{value:.2f} {CURRENCY_LABEL}"


def calculate_items_total(items: list[dict]) -> float:
    total = 0.0
    for item in items:
        price = float(item.get("price", 0.0))
        qty = int(item.get("qty", 0))
        total += price * qty
    return total


def update_marker_trajectory(pid: int, x: float, y: float, angle: float, now: float) -> None:
    path = marker_paths.setdefault(pid, deque())
    path.append((now, x, y, angle))
    cutoff = now - TRAJ_WINDOW_SEC
    while path and path[0][0] < cutoff:
        path.popleft()


def classify_marker_trajectory(pid: int, now: float) -> None:
    path = marker_paths.get(pid)
    if not path or len(path) < TRAJ_MIN_POINTS:
        return

    points = list(path)
    total_dist = 0.0
    turn_count = 0
    angles = [p[3] for p in points]

    for idx in range(1, len(points)):
        _, x0, y0, _ = points[idx - 1]
        _, x1, y1, _ = points[idx]
        total_dist += math.hypot(x1 - x0, y1 - y0)

    x_start, y_start = points[0][1], points[0][2]
    x_end, y_end = points[-1][1], points[-1][2]
    displacement = math.hypot(x_end - x_start, y_end - y_start)
    straightness = displacement / total_dist if total_dist > 0 else 0.0

    for idx in range(1, len(points) - 1):
        _, x0, y0, _ = points[idx - 1]
        _, x1, y1, _ = points[idx]
        _, x2, y2, _ = points[idx + 1]
        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1
        norm1 = math.hypot(v1x, v1y)
        norm2 = math.hypot(v2x, v2y)
        if norm1 == 0 or norm2 == 0:
            continue
        dot = (v1x * v2x + v1y * v2y) / (norm1 * norm2)
        dot = max(-1.0, min(1.0, dot))
        angle_change = math.acos(dot)
        if angle_change > TRAJ_TURN_THRESHOLD:
            turn_count += 1

    angle_range = 0.0
    if angles:
        angle_range = max(angles) - min(angles)

    label = None
    if total_dist < TRAJ_MIN_DISTANCE:
        label = "Rotation" if angle_range > 0.8 else "Stationary"
    elif straightness >= TRAJ_STRAIGHTNESS:
        label = "Linear"
    elif displacement <= TRAJ_LOOP_CLOSE and total_dist >= (TRAJ_MIN_DISTANCE * 2):
        label = "Circular"
    elif turn_count >= 3:
        label = "Zigzag"
    else:
        label = "Freeform"

    last_label = last_traj_label.get(pid)
    last_time = last_traj_time.get(pid, 0.0)
    if label and (label != last_label) and (now - last_time) >= 0.8:
        last_traj_label[pid] = label
        last_traj_time[pid] = now
        log_event(f"Trajectory: Marker {pid} -> {label}")


def load_customer_names() -> list[str]:
    names = []
    with db_session() as conn:
        cursor = conn.execute(
            """
            SELECT name FROM (
                SELECT DISTINCT name AS name FROM faces WHERE role = 'customer'
                UNION
                SELECT DISTINCT customer_name AS name FROM purchases
                UNION
                SELECT DISTINCT customer_name AS name FROM heatmaps
            )
            ORDER BY name
            """
        )
        for (name,) in cursor.fetchall():
            names.append(name)
    return names


 


def record_purchase(customer_name: str, items: dict) -> Tuple[bool, str]:
    if not items:
        return False, "Cart is empty."
    payload = []
    total_qty = 0
    total_price = 0.0
    for pid, data in items.items():
        qty = int(data.get("qty", 0))
        total_qty += qty
        item_name = data.get("name", "Unknown")
        price = data.get("price")
        if price is None:
            item_info = get_inventory_item_by_marker(pid)
            if item_info:
                item_name = item_info["name"]
                price = item_info["price"]
        price = float(price) if price is not None else 0.0
        total_price += price * qty
        payload.append(
            {
                "id": pid,
                "name": item_name,
                "qty": qty,
                "price": price,
            }
        )
    if total_qty <= 0:
        return False, "Cart is empty."

    low_stock = []
    insuff = []
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    updates = []

    with db_session() as conn:
        for pid, data in items.items():
            qty = int(data.get("qty", 0))
            name = data.get("name", "Unknown")
            row = conn.execute(
                "SELECT name, stock FROM inventory WHERE marker_id = ?",
                (pid,),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT name, stock FROM inventory WHERE name = ?",
                    (name,),
                ).fetchone()
            if row is None:
                continue
            db_name, stock = row[0], int(row[1])
            if qty > stock:
                insuff.append((db_name, stock, qty))
            else:
                updates.append((db_name, stock, qty))

        if insuff:
            details = ", ".join(
                f"{name} (available {stock}, requested {qty})"
                for name, stock, qty in insuff
            )
            return False, f"Insufficient stock: {details}."

        for db_name, stock, qty in updates:
            new_stock = max(0, stock - qty)
            conn.execute(
                "UPDATE inventory SET stock = ?, updated_at = ? WHERE name = ?",
                (new_stock, now, db_name),
            )
            if new_stock <= LOW_STOCK_THRESHOLD:
                low_stock.append((db_name, new_stock))

        conn.execute(
            """
            INSERT INTO purchases (customer_name, items_json, total_qty, total_price, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                customer_name,
                json.dumps(payload),
                total_qty,
                total_price,
                now,
            ),
        )

    export_database_sql(DATABASE_SQL_PATH)
    if low_stock:
        lines = ["Low stock alert:"]
        for name, qty in low_stock:
            lines.append(f"- {name}: {qty} left")
        signals.update_admin_display.emit("\n".join(lines))
    return True, "Payment confirmed."


def load_purchase_history(customer_name: str, limit: int = 5) -> list[dict]:
    history = []
    with db_session() as conn:
        cursor = conn.execute(
            """
            SELECT items_json, total_qty, total_price, created_at
            FROM purchases
            WHERE customer_name = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (customer_name, limit),
        )
        for items_json, total_qty, total_price, created_at in cursor.fetchall():
            try:
                items = json.loads(items_json)
            except json.JSONDecodeError:
                items = []
            history.append(
                {
                    "items": items,
                    "total_qty": int(total_qty),
                    "total_price": float(total_price),
                    "created_at": created_at,
                }
            )
    return history


def load_all_purchases(limit: int = 20) -> list[dict]:
    purchases = []
    with db_session() as conn:
        cursor = conn.execute(
            """
            SELECT customer_name, items_json, total_qty, total_price, created_at
            FROM purchases
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        for customer_name, items_json, total_qty, total_price, created_at in cursor.fetchall():
            try:
                items = json.loads(items_json)
            except json.JSONDecodeError:
                items = []
            purchases.append(
                {
                    "customer_name": customer_name,
                    "items": items,
                    "total_qty": int(total_qty),
                    "total_price": float(total_price),
                    "created_at": created_at,
                }
            )
    return purchases


def load_sales_metrics() -> dict:
    with db_session() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(total_qty), 0),
                   COALESCE(SUM(total_price), 0), MAX(created_at)
            FROM purchases
            """
        ).fetchone()
    total_orders = int(row[0]) if row else 0
    total_qty = int(row[1]) if row else 0
    total_price = float(row[2]) if row else 0.0
    latest = row[3] if row and row[3] else "--"
    return {
        "total_orders": total_orders,
        "total_qty": total_qty,
        "total_price": total_price,
        "latest": latest,
    }


def load_purchase_customers() -> list[str]:
    customers = []
    with db_session() as conn:
        cursor = conn.execute(
            "SELECT DISTINCT customer_name FROM purchases ORDER BY customer_name"
        )
        for (name,) in cursor.fetchall():
            customers.append(name)
    return customers


def upsert_heatmap_record(customer_name: str, kind: str, path: Path) -> None:
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO heatmaps (customer_name, kind, image_path, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(customer_name, kind) DO UPDATE SET
                image_path = excluded.image_path,
                updated_at = excluded.updated_at
            """,
            (
                customer_name,
                kind,
                str(path),
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    export_database_sql(DATABASE_SQL_PATH)


def save_gaze_calibration(customer_name: str, feature_map: Optional[np.ndarray]) -> None:
    if not customer_name or feature_map is None:
        return
    payload = json.dumps(feature_map.tolist())
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO gaze_calibration (customer_name, feature_map, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(customer_name) DO UPDATE SET
                feature_map = excluded.feature_map,
                updated_at = excluded.updated_at
            """,
            (
                customer_name,
                payload,
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    export_database_sql(DATABASE_SQL_PATH)


def load_gaze_calibration(customer_name: str) -> Optional[np.ndarray]:
    if not customer_name:
        return None
    with db_session() as conn:
        row = conn.execute(
            "SELECT feature_map FROM gaze_calibration WHERE customer_name = ?",
            (customer_name,),
        ).fetchone()
    if not row or not row[0]:
        return None
    try:
        values = json.loads(row[0])
        arr = np.array(values, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] != 2:
        return None
    return arr


def load_heatmap_path(customer_name: str, kind: str) -> Path:
    with db_session() as conn:
        row = conn.execute(
            """
            SELECT image_path
            FROM heatmaps
            WHERE customer_name = ? AND kind = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (customer_name, kind),
        ).fetchone()
    if row and row[0]:
        return Path(row[0])
    return HEATMAP_DIR / f"{safe_name(customer_name)}_{kind}.png"


def render_heatmap_image(acc: np.ndarray) -> Optional[np.ndarray]:
    if acc is None:
        return None
    if not HEATMAP_BG_PATH.exists():
        return None
    bg = cv2.imread(str(HEATMAP_BG_PATH))
    if bg is None:
        return None

    max_val = float(acc.max()) if acc is not None else 0.0
    heat = None
    target_h, target_w = acc.shape[:2]
    if bg.shape[1] != target_w or bg.shape[0] != target_h:
        bg = cv2.resize(bg, (target_w, target_h))

    if max_val > 0:
        scale_max = max_val
        if HEATMAP_PERCENTILE:
            scale_max = float(np.percentile(acc, HEATMAP_PERCENTILE))
            if scale_max <= 0:
                scale_max = max_val
        heat_norm = np.clip(acc / scale_max, 0.0, 1.0)
        heat_norm = np.power(heat_norm, HEATMAP_GAMMA)
        heat = (heat_norm * 255.0).astype("uint8")
        if HEATMAP_DOWNSCALE < 1.0:
            small_w = max(1, int(target_w * HEATMAP_DOWNSCALE))
            small_h = max(1, int(target_h * HEATMAP_DOWNSCALE))
            heat = cv2.resize(heat, (small_w, small_h), interpolation=cv2.INTER_AREA)
        if HEATMAP_BLUR > 1:
            heat = cv2.GaussianBlur(heat, (HEATMAP_BLUR, HEATMAP_BLUR), 0)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        if heat.shape[1] != target_w or heat.shape[0] != target_h:
            heat = cv2.resize(heat, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    x_norm, y_norm, w_norm, h_norm = HEATMAP_CAMERA_MASK
    x1 = int(max(0, min(bg.shape[1] - 1, x_norm * bg.shape[1])))
    y1 = int(max(0, min(bg.shape[0] - 1, y_norm * bg.shape[0])))
    x2 = int(max(0, min(bg.shape[1], (x_norm + w_norm) * bg.shape[1])))
    y2 = int(max(0, min(bg.shape[0], (y_norm + h_norm) * bg.shape[0])))
    if x2 > x1 and y2 > y1:
        cv2.rectangle(bg, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)

    if heat is None:
        return bg
    overlay = cv2.addWeighted(bg, 0.4, heat, 0.6, 0)
    mask = cv2.cvtColor(heat, cv2.COLOR_BGR2GRAY) > 30
    if mask.any():
        bg[mask] = overlay[mask]
    return bg


def map_gaze_point(gx: float, gy: float) -> tuple[float, float]:
    global gaze_affine, gaze_feature_map, gaze_features
    try:
        if (
            gaze_feature_map is not None
            and gaze_features is not None
            and (time.time() - gaze_features_timestamp) <= CALIB_STABLE_TIMEOUT
        ):
            vec = np.array(
                [
                    gaze_features[0],
                    gaze_features[1],
                    gaze_features[2],
                    gaze_features[3],
                    gaze_features[4],
                    gaze_features[5],
                    1.0,
                ],
                dtype=np.float32,
            )
            sx = float(np.dot(gaze_feature_map[0], vec))
            sy = float(np.dot(gaze_feature_map[1], vec))
            return max(0.0, min(1.0, sx)), max(0.0, min(1.0, sy))
        if gaze_affine is not None:
            mapped = gaze_affine @ np.array([gx, gy, 1.0], dtype=np.float32)
            sx = float(mapped[0])
            sy = float(mapped[1])
            return max(0.0, min(1.0, sx)), max(0.0, min(1.0, sy))
    except Exception:
        return max(0.0, min(1.0, gx)), max(0.0, min(1.0, gy))
    return max(0.0, min(1.0, gx)), max(0.0, min(1.0, gy))


def save_heatmap_image(name: str, suffix: str, acc: np.ndarray, directory: Optional[Path] = None) -> Optional[Path]:
    image = render_heatmap_image(acc)
    if image is None:
        return None
    target_dir = directory or HEATMAP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{safe_name(name)}_{suffix}.png"
    cv2.imwrite(str(path), image)
    upsert_heatmap_record(name, suffix, path)
    return path


def update_average_heatmap(name: str, acc: np.ndarray) -> Optional[Path]:
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    avg_path = HEATMAP_DATA_DIR / f"{safe_name(name)}_average.npz"
    if avg_path.exists():
        data = np.load(str(avg_path))
        avg_acc = data["accum"]
        count = int(data["count"])
    else:
        avg_acc = np.zeros_like(acc, dtype=np.float32)
        count = 0

    if avg_acc.shape != acc.shape:
        avg_acc = np.zeros_like(acc, dtype=np.float32)
        count = 0

    avg_acc = avg_acc + acc.astype(np.float32)
    count += 1
    np.savez_compressed(str(avg_path), accum=avg_acc, count=count)
    return save_heatmap_image(
        name,
        "average",
        avg_acc / max(count, 1),
        directory=HEATMAP_DATA_DIR,
    )


def start_customer_session(name: str) -> None:
    global active_customer_name, session_active, session_heatmap_acc, heatmap_target_size
    with session_lock:
        active_customer_name = name
        session_active = True
        session_heatmap_acc = None
        if HEATMAP_BG_PATH.exists():
            bg = cv2.imread(str(HEATMAP_BG_PATH))
            if bg is not None:
                heatmap_target_size = (bg.shape[0], bg.shape[1])


def finalize_customer_session() -> None:
    global active_customer_name, session_active, session_heatmap_acc
    with session_lock:
        name = active_customer_name
        acc = session_heatmap_acc
        session_active = False
        active_customer_name = None
        session_heatmap_acc = None
    if name and acc is not None:
        save_heatmap_image(name, "latest", acc)
        update_average_heatmap(name, acc)


 


def compose_face_status_text() -> str:
    with face_status_lock:
        name = current_identity["name"]
        emotion = current_emotion
    return f"Face: {name} | Emotion: {emotion}"


def update_face_status_text() -> None:
    signals.update_face_status.emit(compose_face_status_text())


def add_shadow(widget: QtWidgets.QWidget, blur: int = 18, alpha: int = 60) -> None:
    shadow = QtWidgets.QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setOffset(0, 6)
    shadow.setColor(QtGui.QColor(0, 0, 0, alpha))
    widget.setGraphicsEffect(shadow)


def build_card(title: str, content: QtWidgets.QWidget, role: str = "card") -> Tuple[QtWidgets.QFrame, QtWidgets.QLabel]:
    frame = QtWidgets.QFrame()
    frame.setProperty("role", role)
    layout = QtWidgets.QVBoxLayout(frame)
    layout.setContentsMargins(18, 16, 18, 16)
    layout.setSpacing(10)

    title_label = QtWidgets.QLabel(title)
    title_label.setProperty("role", "card-title")
    layout.addWidget(title_label)
    layout.addWidget(content)
    add_shadow(frame)
    return frame, title_label


class UiSignals(QtCore.QObject):
    set_ui_state = QtCore.Signal(str)
    set_menu_context = QtCore.Signal(str)
    draw_ar_object = QtCore.Signal(int, str, float, float, float)
    remove_ar_object = QtCore.Signal(int)
    rotate_right = QtCore.Signal(int)
    rotate_left = QtCore.Signal(int)
    handle_customer_selection = QtCore.Signal()
    handle_admin_selection = QtCore.Signal()
    handle_manage_users_selection = QtCore.Signal()
    handle_edit_stock_selection = QtCore.Signal()
    handle_purchases_selection = QtCore.Signal()
    handle_inventory_selection = QtCore.Signal()
    handle_reports_selection = QtCore.Signal()
    show_payment_state = QtCore.Signal(str, bool)
    update_face_status = QtCore.Signal(str)
    update_auth_error = QtCore.Signal(str)
    update_customer_display = QtCore.Signal(str)
    update_admin_display = QtCore.Signal(str)
    camera_frame = QtCore.Signal(object)
    capture_calibration = QtCore.Signal()


signals = UiSignals()


class CircularMenuWidget(QtWidgets.QWidget):
    activated = QtCore.Signal()

    def __init__(self, options: list[str], parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.options = options
        self.selected_index = 0
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def set_options(self, options: list[str]) -> None:
        self.options = options
        self.selected_index = 0
        self.update()

    def set_selected_index(self, index: int) -> None:
        if not self.options:
            return
        self.selected_index = index % len(self.options)
        self.update()

    def select_next(self) -> None:
        if not self.options:
            return
        self.selected_index = (self.selected_index + 1) % len(self.options)
        self.update()

    def select_prev(self) -> None:
        if not self.options:
            return
        self.selected_index = (self.selected_index - 1) % len(self.options)
        self.update()

    def _compute_menu_geometry(self, rect: QtCore.QRect, count: int) -> tuple[float, float, float]:
        edge_pad = 14
        outer_radius = max((min(rect.width(), rect.height()) / 2) - edge_pad, 80)
        center_radius = 30
        min_gap = 10 if count <= 6 else 8
        bubble_radius = min(60, max(30, outer_radius * 0.38))

        ring_radius = None
        while bubble_radius >= 24:
            min_ring = (count * (2 * bubble_radius + min_gap)) / (2 * math.pi)
            min_ring = max(min_ring, center_radius + bubble_radius + 10)
            if min_ring <= (outer_radius - bubble_radius):
                ring_radius = min_ring
                break
            bubble_radius -= 2

        if ring_radius is None:
            bubble_radius = max(24, bubble_radius)
            ring_radius = max(center_radius + bubble_radius + 10, outer_radius - bubble_radius)

        return ring_radius, bubble_radius, center_radius

    def _bubble_positions(self, rect: QtCore.QRect, count: int, radius: float) -> list[QtCore.QPointF]:
        center = rect.center()
        positions = []
        for idx in range(count):
            angle = (2 * math.pi * idx / count) - (math.pi / 2)
            x = center.x() + radius * math.cos(angle)
            y = center.y() + radius * math.sin(angle)
            positions.append(QtCore.QPointF(x, y))
        return positions

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        if not painter.begin(self):
            return
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        rect = self.rect()
        count = max(len(self.options), 1)
        radius, bubble_radius, center_radius = self._compute_menu_geometry(rect, count)
        center = rect.center()

        ring_color = QtGui.QColor(BORDER)
        ring_pen = QtGui.QPen(ring_color, 1)
        painter.setPen(ring_pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(center, radius + 8, radius + 8)

        for idx, label in enumerate(self.options):
            angle = (2 * math.pi * idx / count) - (math.pi / 2)
            x = center.x() + radius * math.cos(angle)
            y = center.y() + radius * math.sin(angle)
            is_selected = idx == self.selected_index

            bubble_color = QtGui.QColor(CARD)
            if is_selected:
                bubble_color = QtGui.QColor(ACCENT)
                bubble_color.setAlpha(35)
            text_color = QtGui.QColor(ACCENT if is_selected else MUTED)

            painter.setPen(QtGui.QPen(QtGui.QColor(BORDER), 1))
            painter.setBrush(bubble_color)
            painter.drawEllipse(QtCore.QPointF(x, y), bubble_radius, bubble_radius)

            painter.setPen(text_color)
            font = painter.font()
            font_size = max(8, min(11, int(bubble_radius / 4.4)))
            if len(label) > 16:
                font_size -= 1
            if len(label) > 22:
                font_size -= 1
            font.setPointSize(font_size)
            font.setBold(True)
            painter.setFont(font)
            text_rect = QtCore.QRectF(
                x - bubble_radius + 6,
                y - bubble_radius + 6,
                bubble_radius * 2 - 12,
                bubble_radius * 2 - 12,
            )
            painter.drawText(text_rect, QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, label)

        painter.setPen(QtGui.QPen(QtGui.QColor(BORDER), 1))
        painter.setBrush(QtGui.QColor(CARD_SOFT))
        painter.drawEllipse(center, center_radius, center_radius)
        painter.setPen(QtGui.QColor(MUTED))
        painter.drawText(
            QtCore.QRectF(center.x() - 24, center.y() - 10, 48, 20),
            QtCore.Qt.AlignCenter,
            "Select",
        )
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self.options:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        self.setFocus(QtCore.Qt.MouseFocusReason)
        rect = self.rect()
        count = len(self.options)
        radius, bubble_radius, center_radius = self._compute_menu_geometry(rect, count)
        center = rect.center()
        click_pos = event.position()

        if QtCore.QLineF(click_pos, QtCore.QPointF(center)).length() <= center_radius:
            self.activated.emit()
            return

        hit_radius = bubble_radius + 6
        positions = self._bubble_positions(rect, count, radius)
        for idx, pos in enumerate(positions):
            if QtCore.QLineF(click_pos, pos).length() <= hit_radius:
                if idx == self.selected_index:
                    self.activated.emit()
                else:
                    self.selected_index = idx
                    self.update()
                return

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key in (QtCore.Qt.Key_Left, QtCore.Qt.Key_Up):
            self.select_prev()
            return
        if key in (QtCore.Qt.Key_Right, QtCore.Qt.Key_Down):
            self.select_next()
            return
        if key in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter, QtCore.Qt.Key_Space):
            self.activated.emit()
            return
        super().keyPressEvent(event)


class AROverlayWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(320)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self._image_cache: dict[tuple[int, str], Optional[QtGui.QPixmap]] = {}

    def _candidate_image_paths(self, name: str, pid: int) -> list[Path]:
        candidates = []
        if name:
            safe = "".join(ch if ch.isalnum() or ch in " _-" else "" for ch in name).strip()
            variants = [
                name,
                safe,
                safe.replace(" ", "_"),
                safe.lower(),
                safe.lower().replace(" ", "_"),
            ]
            for variant in variants:
                if variant:
                    candidates.append(variant)
        candidates.append(str(pid))
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        paths = []
        for base in candidates:
            for ext in exts:
                paths.append(IMAGE_DIR / f"{base}{ext}")
        return paths

    def _load_product_pixmap(self, pid: int, name: str) -> Optional[QtGui.QPixmap]:
        key = (pid, name or "")
        if key in self._image_cache:
            return self._image_cache[key]
        for path in self._candidate_image_paths(name, pid):
            if path.exists():
                pixmap = QtGui.QPixmap(str(path))
                if not pixmap.isNull():
                    self._image_cache[key] = pixmap
                    return pixmap
        self._image_cache[key] = None
        return None

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter()
        if not painter.begin(self):
            return
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor(CARD_SOFT))

        if not ar_objects:
            painter.setPen(QtGui.QColor(MUTED))
            painter.drawText(rect, QtCore.Qt.AlignCenter, "Awaiting marker input...")
            painter.end()
            return

        for pid, data in ar_objects.items():
            x = data["x"] * rect.width()
            y = data["y"] * rect.height()
            angle = data["angle"]
            name = data["name"]

            pixmap = self._load_product_pixmap(pid, name)
            if pixmap is not None:
                base_size = max(90, min(160, int(min(rect.width(), rect.height()) * 0.22)))
                half = base_size / 2
                left = max(rect.left() + 8, min(x - half, rect.right() - base_size - 8))
                top = max(rect.top() + 8, min(y - half, rect.bottom() - base_size - 8))

                bg_rect = QtCore.QRectF(left, top, base_size, base_size)
                painter.setPen(QtGui.QPen(QtGui.QColor(BORDER), 1))
                painter.setBrush(QtGui.QColor(CARD))
                painter.drawRoundedRect(bg_rect, 12, 12)

                scaled = pixmap.scaled(
                    int(base_size * 0.82),
                    int(base_size * 0.82),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation,
                )
                img_left = left + (base_size - scaled.width()) / 2
                img_top = top + (base_size - scaled.height()) / 2
                painter.drawPixmap(QtCore.QPointF(img_left, img_top), scaled)
            else:
                ring_color = QtGui.QColor(ACCENT)
                ring_color.setAlpha(110)
                glow_pen = QtGui.QPen(ring_color, 1)
                painter.setPen(glow_pen)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.drawEllipse(QtCore.QPointF(x, y), 56, 56)

                body_color = QtGui.QColor(ACCENT)
                body_color.setAlpha(26)
                outline_color = QtGui.QColor(ACCENT)
                outline_color.setAlpha(140)
                painter.setPen(QtGui.QPen(outline_color, 1))
                painter.setBrush(body_color)
                painter.drawEllipse(QtCore.QPointF(x, y), 48, 48)

                painter.setPen(QtGui.QColor(TEXT))
                font = painter.font()
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(
                    QtCore.QRectF(x - 50, y - 12, 100, 24),
                    QtCore.Qt.AlignCenter,
                    name,
                )

            arrow_len = 70
            end_x = x + arrow_len * math.cos(angle)
            end_y = y + arrow_len * math.sin(angle)
            arrow_color = QtGui.QColor(ACCENT)
            arrow_color.setAlpha(160)
            arrow_pen = QtGui.QPen(arrow_color, 2)
            painter.setPen(arrow_pen)
            painter.drawLine(QtCore.QPointF(x, y), QtCore.QPointF(end_x, end_y))

            head_len = 10
            head_angle = math.radians(25)
            left_x = end_x - head_len * math.cos(angle - head_angle)
            left_y = end_y - head_len * math.sin(angle - head_angle)
            right_x = end_x - head_len * math.cos(angle + head_angle)
            right_y = end_y - head_len * math.sin(angle + head_angle)
            painter.drawLine(QtCore.QPointF(end_x, end_y), QtCore.QPointF(left_x, left_y))
            painter.drawLine(QtCore.QPointF(end_x, end_y), QtCore.QPointF(right_x, right_y))
        painter.end()


class GazeDotOverlay(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._pos: Optional[tuple[float, float]] = None
        self._radius = 22
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_position(self, norm_x: Optional[float], norm_y: Optional[float]) -> None:
        if norm_x is None or norm_y is None:
            self._pos = None
        else:
            self._pos = (float(norm_x), float(norm_y))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if self._pos is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect()
        x = rect.left() + self._pos[0] * rect.width()
        y = rect.top() + self._pos[1] * rect.height()
        dot_color = QtGui.QColor(ACCENT)
        dot_color.setAlpha(210)
        outline = QtGui.QColor(255, 255, 255, 220)
        painter.setPen(QtGui.QPen(outline, 2))
        painter.setBrush(dot_color)
        painter.drawEllipse(QtCore.QPointF(x, y), self._radius, self._radius)
        painter.end()


class CalibrationTargetOverlay(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._target: Optional[tuple[float, float]] = None
        self._label: str = ""
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground, True)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    def set_target(self, norm_x: Optional[float], norm_y: Optional[float], label: str = "") -> None:
        if norm_x is None or norm_y is None:
            self._target = None
            self._label = ""
        else:
            self._target = (float(norm_x), float(norm_y))
            self._label = label
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if self._target is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect()
        x = rect.left() + self._target[0] * rect.width()
        y = rect.top() + self._target[1] * rect.height()
        ring_color = QtGui.QColor(ACCENT)
        ring_color.setAlpha(200)
        painter.setPen(QtGui.QPen(ring_color, 2))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawEllipse(QtCore.QPointF(x, y), 12, 12)
        painter.setBrush(ring_color)
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawEllipse(QtCore.QPointF(x, y), 4, 4)
        if self._label:
            painter.setPen(QtGui.QColor(TEXT))
            font = painter.font()
            font.setPointSize(10)
            painter.setFont(font)
            painter.drawText(QtCore.QPointF(x + 14, y - 10), self._label)
        painter.end()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Smart Pharmacy")
        self.setMinimumSize(1250, 820)
        self._last_camera_frame = None
        self._text_boxes = []

        central = QtWidgets.QWidget()
        central.setObjectName("app")
        self.setCentralWidget(central)

        root_layout = QtWidgets.QVBoxLayout(central)
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(12)

        self.gaze_overlay = GazeDotOverlay(central)
        self.calibration_overlay = CalibrationTargetOverlay(central)
        self._update_gaze_overlay_geometry()
        self.gaze_overlay.raise_()
        self.calibration_overlay.raise_()

        header = self._build_header()
        root_layout.addWidget(header)

        self.state_stack = QtWidgets.QStackedWidget()
        self.idle_page = self._build_idle_page()
        self.customer_page = self._build_customer_page()
        self.admin_page = self._build_admin_page()
        self.payment_page = self._build_payment_page()
        self.manage_users_page = self._build_manage_users_page()
        self.edit_stock_page = self._build_edit_stock_page()
        self.purchases_page = self._build_purchases_page()
        self.inventory_page = self._build_inventory_page()
        self.reports_page = self._build_reports_page()

        self.state_stack.addWidget(self.idle_page)
        self.state_stack.addWidget(self.customer_page)
        self.state_stack.addWidget(self.admin_page)
        self.state_stack.addWidget(self.payment_page)
        self.state_stack.addWidget(self.manage_users_page)
        self.state_stack.addWidget(self.edit_stock_page)
        self.state_stack.addWidget(self.purchases_page)
        self.state_stack.addWidget(self.inventory_page)
        self.state_stack.addWidget(self.reports_page)

        root_layout.addWidget(self.state_stack, 1)

        self._connect_signals()
        self.customer_menu_widget.activated.connect(self.handle_customer_selection)
        self.admin_menu_widget.activated.connect(self.handle_admin_selection)
        self.manage_users_menu.activated.connect(self.handle_manage_users_selection)
        self.edit_stock_menu.activated.connect(self.handle_edit_stock_selection)
        self.purchases_menu.activated.connect(self.handle_purchases_selection)
        self.inventory_menu.activated.connect(self.handle_inventory_selection)
        self.reports_menu.activated.connect(self.handle_reports_selection)

        self.cart_timer = QtCore.QTimer(self)
        self.cart_timer.timeout.connect(self.update_cart_display)
        self.cart_timer.start(600)

        self.admin_metrics_timer = QtCore.QTimer(self)
        self.admin_metrics_timer.timeout.connect(self.refresh_admin_metrics)
        self.admin_metrics_timer.start(1200)

        self.gaze_scroll_dir = None
        self.gaze_scroll_start = 0.0
        self.gaze_scroll_last = 0.0
        self.gaze_scroll_timer = QtCore.QTimer(self)
        self.gaze_scroll_timer.timeout.connect(self.update_gaze_scroll)
        self.gaze_scroll_timer.start(120)

        self.gaze_overlay_timer = QtCore.QTimer(self)
        self.gaze_overlay_timer.timeout.connect(self.update_gaze_overlay)
        self.gaze_overlay_timer.start(80)

        self.calibration_timer = QtCore.QTimer(self)
        self.calibration_timer.timeout.connect(self.update_calibration_progress)
        self.calibration_timer.start(80)

        self.update_face_status(compose_face_status_text())
        self.set_ui_state("idle")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_gaze_overlay_geometry()

    def _update_gaze_overlay_geometry(self) -> None:
        if not hasattr(self, "gaze_overlay"):
            return
        central = self.centralWidget()
        if central is None:
            return
        self.gaze_overlay.setGeometry(central.rect())
        self.gaze_overlay.raise_()
        if hasattr(self, "calibration_overlay"):
            self.calibration_overlay.setGeometry(central.rect())
            self.calibration_overlay.raise_()

    def _build_header(self) -> QtWidgets.QWidget:
        header = QtWidgets.QFrame()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(16)

        brand_col = QtWidgets.QVBoxLayout()
        brand_col.setSpacing(10)
        title = QtWidgets.QLabel("Smart Pharmacy")
        title.setProperty("role", "title")
        subtitle = QtWidgets.QLabel(
            "Welcome to the TUIO smart pharmacy 'powered by HCI'"
        )
        subtitle.setProperty("role", "subtitle")
        subtitle.setWordWrap(True)
        self.state_badge = QtWidgets.QLabel("STATE: IDLE")
        self.state_badge.setProperty("role", "badge")
        self.auth_error = QtWidgets.QLabel("")
        self.auth_error.setProperty("role", "error")
        self.auth_error.setWordWrap(True)
        self.auth_error.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.auth_error.setMinimumHeight(34)
        self.auth_error.setVisible(False)
        subtitle.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.auth_error.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        controls_card = self._build_controls_card()
        brand_col.addWidget(title)
        brand_col.addWidget(subtitle)
        brand_col.addWidget(self.state_badge, alignment=QtCore.Qt.AlignLeft)
        brand_col.addWidget(self.auth_error, alignment=QtCore.Qt.AlignLeft)
        brand_col.addWidget(controls_card, alignment=QtCore.Qt.AlignLeft)
        brand_col.addStretch()

        webcam_card = self._build_webcam_card()

        header_layout.addLayout(brand_col, 2)
        header_layout.addWidget(webcam_card, 1)
        return header

    def _build_controls_card(self) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setProperty("role", "soft-card")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(2)

        controls = QtWidgets.QLabel(
            "Controls: Menu rotate 10 | Select 9 | Products 0-3\n"
            "Pay 11 | Mouse/Enter"
        )
        controls.setProperty("role", "controls")
        controls.setWordWrap(True)
        controls.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        card_layout.addWidget(controls)
        return card

    def _build_webcam_card(self) -> QtWidgets.QFrame:
        preview = QtWidgets.QFrame()
        preview.setProperty("role", "preview")
        preview_layout = QtWidgets.QVBoxLayout(preview)
        preview_layout.setContentsMargins(12, 12, 12, 12)
        preview_layout.setSpacing(8)

        self.preview_label = QtWidgets.QLabel("Webcam preview (initializing)")
        self.preview_label.setProperty("role", "muted")
        self.preview_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_label.setMinimumHeight(140)
        self.preview_label.setScaledContents(False)
        preview_layout.addWidget(self.preview_label)

        self.deepface_status = QtWidgets.QLabel("Face: -- | Emotion: --")
        self.deepface_status.setProperty("role", "status")
        self.deepface_status.setWordWrap(True)
        self.deepface_status.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.deepface_status.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding
        )
        preview_layout.addWidget(self.deepface_status)

        webcam_card, _ = build_card("Camera Feed", preview)
        webcam_card.setMinimumWidth(320)
        return webcam_card

    def _build_idle_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        layout.addStretch()

        card = QtWidgets.QFrame()
        card.setProperty("role", "card")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(26, 22, 26, 22)
        card_layout.setSpacing(8)
        card_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)

        headline = QtWidgets.QLabel("Recognizing user...")
        headline.setProperty("role", "hero")
        detail = QtWidgets.QLabel("DeepFace is active. Face login routes to customer or admin.")
        detail.setProperty("role", "muted")
        detail.setWordWrap(True)
        detail.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        detail.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding)
        line_height = QtGui.QFontMetrics(detail.font()).lineSpacing()
        detail.setMinimumHeight(line_height * 3)

        card_layout.addWidget(headline)
        card_layout.addWidget(detail)
        add_shadow(card)

        layout.addWidget(card, alignment=QtCore.Qt.AlignHCenter)
        layout.addStretch()
        return page

    def _build_customer_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        stack = QtWidgets.QStackedLayout(page)
        stack.setStackingMode(QtWidgets.QStackedLayout.StackAll)

        base = QtWidgets.QWidget()
        base_layout = QtWidgets.QHBoxLayout(base)
        base_layout.setContentsMargins(0, 0, 0, 0)
        base_layout.setSpacing(24)

        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(16)

        badges_row = QtWidgets.QHBoxLayout()
        badges_row.setSpacing(10)
        self.customer_badges = []
        for label in ("Session Active", "Cart Ready", "TUIO Connected", "Heatmaps Active"):
            badge = QtWidgets.QLabel(label)
            badge.setProperty("role", "badge")
            badges_row.addWidget(badge)
            self.customer_badges.append(badge)
        badges_row.addStretch()
        left_col.addLayout(badges_row)

        metrics_row = QtWidgets.QHBoxLayout()
        metrics_row.setSpacing(12)
        lists_row = QtWidgets.QHBoxLayout()
        lists_row.setSpacing(12)

        self.customer_cart_kpi = QtWidgets.QLabel("0 Items")
        self.customer_cart_kpi.setProperty("role", "metric")
        self.customer_total_kpi = QtWidgets.QLabel(format_price(0.0))
        self.customer_total_kpi.setProperty("role", "metric")
        metrics_row.addWidget(self._build_kpi_card("Cart", self.customer_cart_kpi))
        metrics_row.addWidget(self._build_kpi_card("Subtotal", self.customer_total_kpi))

        self.cart_text = QtWidgets.QPlainTextEdit()
        self.cart_text.setReadOnly(True)
        self._configure_text_box(self.cart_text, min_lines=6)
        self.cart_card, _ = build_card("Cart", self.cart_text, role="soft-card")

        self.customer_summary = QtWidgets.QLabel(
            "Subtotal: --\nItems: 0\nLast action: --"
        )
        self.customer_summary.setProperty("role", "muted")
        self.customer_summary.setWordWrap(True)
        self.summary_card = self._build_list_card("Summary", self.customer_summary)

        self.customer_display = QtWidgets.QPlainTextEdit()
        self.customer_display.setReadOnly(True)
        self._configure_text_box(self.customer_display, min_lines=4, auto_fit=False, expand=True)
        self.customer_display.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.customer_display.setPlainText("Awaiting marker input...\n")
        self.status_card, _ = build_card("Status", self.customer_display, role="soft-card")

        self.status_card.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        left_lists_col = QtWidgets.QVBoxLayout()
        left_lists_col.setSpacing(12)
        left_lists_col.addWidget(self.cart_card)
        left_lists_col.addWidget(self.summary_card)

        lists_row.addLayout(left_lists_col, 1)
        lists_row.addWidget(self.status_card, 1)

        left_col.addLayout(metrics_row)
        left_col.addLayout(lists_row)

        self.customer_menu_widget = CircularMenuWidget(customer_menu_options)
        self.customer_menu_card, self.customer_menu_title = build_card(
            "Customer Menu", self.customer_menu_widget, role="soft-card"
        )
        self.customer_menu_widget.setMinimumSize(320, 320)
        self.customer_menu_card.setMinimumWidth(360)

        self.ar_canvas = AROverlayWidget()
        self.ar_card, _ = build_card("AR Overlay", self.ar_canvas, role="card")

        right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QHBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(16)
        right_layout.addWidget(self.ar_card, 2)
        right_layout.addWidget(self.customer_menu_card, 1)

        base_layout.addLayout(left_col, 2)
        base_layout.addWidget(right_container, 2)

        stack.addWidget(base)
        self.customer_menu_overlay = self.customer_menu_card
        return page

    def _build_admin_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        stack = QtWidgets.QStackedLayout(page)
        stack.setStackingMode(QtWidgets.QStackedLayout.StackAll)

        base = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(base)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(16)

        metrics_row = QtWidgets.QHBoxLayout()
        metrics_row.setSpacing(12)
        metrics_list_row = QtWidgets.QHBoxLayout()
        metrics_list_row.setSpacing(12)

        self.inventory_kpi = QtWidgets.QLabel("0 Items")
        self.inventory_kpi.setProperty("role", "metric")
        self.inventory_list = QtWidgets.QLabel("Low stock: 0\nTotal units: 0\nRestock: --")
        self.inventory_list.setProperty("role", "muted")
        self.inventory_list.setWordWrap(True)

        self.sales_kpi = QtWidgets.QLabel("0 Orders")
        self.sales_kpi.setProperty("role", "metric")
        self.sales_list = QtWidgets.QLabel("Items sold: 0\nLatest: --\nRevenue: --")
        self.sales_list.setProperty("role", "muted")
        self.sales_list.setWordWrap(True)

        self.activity_kpi = QtWidgets.QLabel("No Users")
        self.activity_kpi.setProperty("role", "metric")
        self.activity_list = QtWidgets.QLabel("State: idle\nHeatmaps: 0\nLast action: --")
        self.activity_list.setProperty("role", "muted")
        self.activity_list.setWordWrap(True)

        metrics_row.addWidget(self._build_kpi_card("Inventory", self.inventory_kpi))
        metrics_row.addWidget(self._build_kpi_card("Sales", self.sales_kpi))
        metrics_row.addWidget(self._build_kpi_card("Activity", self.activity_kpi))

        metrics_list_row.addWidget(self._build_list_card("Inventory", self.inventory_list))
        metrics_list_row.addWidget(self._build_list_card("Sales", self.sales_list))
        metrics_list_row.addWidget(self._build_list_card("Activity", self.activity_list))

        badges_row = QtWidgets.QHBoxLayout()
        badges_row.setSpacing(10)
        self.status_badges = []
        for label in ("System Online", "Camera OK", "TUIO Connected", "Heatmaps Active"):
            badge = QtWidgets.QLabel(label)
            badge.setProperty("role", "badge")
            badges_row.addWidget(badge)
            self.status_badges.append(badge)
        badges_row.addStretch()
        left_col.addLayout(badges_row)

        left_col.addLayout(metrics_row)
        left_col.addLayout(metrics_list_row)

        self.admin_display = QtWidgets.QPlainTextEdit()
        self.admin_display.setReadOnly(True)
        self._configure_text_box(self.admin_display, min_lines=2)
        self.admin_display.setPlainText(
            "Admin status updates appear here.\nUse marker 10 to navigate menus and marker 9 to select."
        )
        admin_display_card, _ = build_card("Admin Status", self.admin_display, role="soft-card")
        admin_display_card.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed
        )
        left_col.addWidget(admin_display_card)

        self.admin_inventory_text = QtWidgets.QPlainTextEdit()
        self.admin_inventory_text.setReadOnly(True)
        self._configure_text_box(self.admin_inventory_text, min_lines=6)
        self.admin_inventory_text.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.admin_inventory_text.setPlainText("No inventory loaded.")
        self.admin_inventory_card, _ = build_card(
            "Inventory List", self.admin_inventory_text, role="card"
        )
        self.admin_inventory_card.hide()
        left_col.addWidget(self.admin_inventory_card)

        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(16)

        self.admin_menu_widget = CircularMenuWidget(admin_menu_options)
        self.admin_menu_card, self.admin_menu_title = build_card(
            "Admin Menu", self.admin_menu_widget, role="soft-card"
        )
        right_col.addWidget(self.admin_menu_card)
        right_col.addStretch()

        layout.addLayout(left_col, 3)
        layout.addLayout(right_col, 1)

        stack.addWidget(base)
        self.admin_menu_overlay = self.admin_menu_card
        return page

    def _build_payment_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)
        layout.addStretch()

        card = QtWidgets.QFrame()
        card.setProperty("role", "card")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(28, 24, 28, 24)
        card_layout.setSpacing(8)

        headline = QtWidgets.QLabel("Payment")
        headline.setProperty("role", "hero")
        self.payment_status = QtWidgets.QLabel("Waiting for payment marker (11).")
        self.payment_status.setProperty("role", "muted")
        self.payment_status.setWordWrap(True)
        self.payment_status.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.payment_status.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.MinimumExpanding
        )
        line_height = QtGui.QFontMetrics(self.payment_status.font()).lineSpacing()
        self.payment_status.setMinimumHeight(line_height * 2)
        card_layout.addWidget(headline)
        card_layout.addWidget(self.payment_status)
        add_shadow(card)

        layout.addWidget(card, alignment=QtCore.Qt.AlignHCenter)
        layout.addStretch()
        return page

    def _build_manage_users_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        card = QtWidgets.QFrame()
        card.setProperty("role", "card")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 22)
        card_layout.setSpacing(12)

        headline = QtWidgets.QLabel("Manage Users")
        headline.setProperty("role", "hero")
        instructions = QtWidgets.QLabel(
            "Rotate marker 10 to choose a user or action, then select with marker 9."
        )
        instructions.setProperty("role", "muted")
        instructions.setWordWrap(True)

        self.manage_users_menu = CircularMenuWidget([])
        self.manage_users_menu_card, self.manage_users_menu_title = build_card(
            "Users", self.manage_users_menu, role="soft-card"
        )

        self.manage_users_status = QtWidgets.QLabel("Select a user.")
        self.manage_users_status.setProperty("role", "muted")
        self.manage_users_status.setWordWrap(True)

        card_layout.addWidget(headline)
        card_layout.addWidget(instructions)
        card_layout.addWidget(self.manage_users_menu_card)
        card_layout.addWidget(self.manage_users_status)
        add_shadow(card)

        layout.addStretch()
        layout.addWidget(card, alignment=QtCore.Qt.AlignHCenter)
        layout.addStretch()
        return page

    def _build_edit_stock_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(20)

        card = QtWidgets.QFrame()
        card.setProperty("role", "card")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(24, 22, 24, 22)
        card_layout.setSpacing(12)

        headline = QtWidgets.QLabel("Edit Stock")
        headline.setProperty("role", "hero")
        instructions = QtWidgets.QLabel(
            "Rotate marker 10 to choose a medicine or action, then select with marker 9."
        )
        instructions.setProperty("role", "muted")
        instructions.setWordWrap(True)

        self.edit_stock_menu = CircularMenuWidget([])
        self.edit_stock_menu_card, self.edit_stock_menu_title = build_card(
            "Medicines", self.edit_stock_menu, role="soft-card"
        )

        self.edit_stock_status = QtWidgets.QLabel("Select a medicine.")
        self.edit_stock_status.setProperty("role", "muted")
        self.edit_stock_status.setWordWrap(True)

        card_layout.addWidget(headline)
        card_layout.addWidget(instructions)
        card_layout.addWidget(self.edit_stock_menu_card)
        card_layout.addWidget(self.edit_stock_status)
        add_shadow(card)

        layout.addStretch()
        layout.addWidget(card, alignment=QtCore.Qt.AlignHCenter)
        layout.addStretch()
        return page

    def _build_purchases_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(16)

        headline = QtWidgets.QLabel("Purchase History")
        headline.setProperty("role", "hero")
        instructions = QtWidgets.QLabel(
            "Rotate marker 10 to choose a customer, then select with marker 9."
        )
        instructions.setProperty("role", "muted")
        instructions.setWordWrap(True)

        self.purchases_menu = CircularMenuWidget([])
        self.purchases_menu_card, self.purchases_menu_title = build_card(
            "Customers", self.purchases_menu, role="soft-card"
        )

        self.purchases_status = QtWidgets.QLabel("Select a customer.")
        self.purchases_status.setProperty("role", "muted")
        self.purchases_status.setWordWrap(True)

        left_col.addWidget(headline)
        left_col.addWidget(instructions)
        left_col.addWidget(self.purchases_menu_card)
        left_col.addWidget(self.purchases_status)
        left_col.addStretch()

        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(16)

        self.purchases_text = QtWidgets.QPlainTextEdit()
        self.purchases_text.setReadOnly(True)
        self._configure_text_box(self.purchases_text, min_lines=8, auto_fit=False, expand=True)
        self.purchases_text.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.purchases_text.setPlainText("No customer selected.")
        purchases_card, _ = build_card("Purchases", self.purchases_text, role="card")
        purchases_card.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        right_col.addWidget(purchases_card, 1)

        layout.addLayout(left_col, 1)
        layout.addLayout(right_col, 2)
        return page

    def _build_inventory_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(16)

        headline = QtWidgets.QLabel("View Inventory")
        headline.setProperty("role", "hero")
        instructions = QtWidgets.QLabel(
            "Rotate marker 10 to choose a medicine, then select with marker 9."
        )
        instructions.setProperty("role", "muted")
        instructions.setWordWrap(True)

        self.inventory_menu = CircularMenuWidget([])
        self.inventory_menu_card, self.inventory_menu_title = build_card(
            "Medicines", self.inventory_menu, role="soft-card"
        )

        self.inventory_status = QtWidgets.QLabel("Select a medicine.")
        self.inventory_status.setProperty("role", "muted")
        self.inventory_status.setWordWrap(True)

        left_col.addWidget(headline)
        left_col.addWidget(instructions)
        left_col.addWidget(self.inventory_menu_card)
        left_col.addWidget(self.inventory_status)
        left_col.addStretch()

        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(16)

        self.inventory_text = QtWidgets.QPlainTextEdit()
        self.inventory_text.setReadOnly(True)
        self._configure_text_box(self.inventory_text, min_lines=8, auto_fit=False, expand=True)
        self.inventory_text.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.inventory_text.setPlainText("No medicine selected.")
        inventory_card, _ = build_card("Inventory Details", self.inventory_text, role="card")
        inventory_card.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        right_col.addWidget(inventory_card, 1)

        layout.addLayout(left_col, 1)
        layout.addLayout(right_col, 2)
        return page

    def _build_reports_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(24)

        left_col = QtWidgets.QVBoxLayout()
        left_col.setSpacing(16)
        left_col.addStretch()

        self.reports_menu = CircularMenuWidget([])
        self.reports_menu_card, self.reports_menu_title = build_card(
            "Customers", self.reports_menu, role="soft-card"
        )
        left_col.addWidget(self.reports_menu_card)
        left_col.addStretch()

        right_col = QtWidgets.QVBoxLayout()
        right_col.setSpacing(16)

        self.reports_image = QtWidgets.QLabel("Select a customer to view heatmap.")
        self.reports_image.setAlignment(QtCore.Qt.AlignCenter)
        self.reports_image.setMinimumSize(520, 320)
        self.reports_image.setWordWrap(True)
        self.reports_image.setProperty("role", "muted")
        self.reports_image_card, _ = build_card("Heatmap", self.reports_image, role="card")

        self.reports_caption = QtWidgets.QLabel("")
        self.reports_caption.setProperty("role", "muted")
        self.reports_caption.setWordWrap(True)

        right_col.addWidget(self.reports_image_card, 2)
        right_col.addWidget(self.reports_caption)
        right_col.addStretch()

        layout.addLayout(left_col, 1)
        layout.addLayout(right_col, 2)
        return page

    def _build_metric_card(
        self, title: str, kpi_label: QtWidgets.QLabel, list_label: QtWidgets.QLabel
    ) -> QtWidgets.QFrame:
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        content_layout.addWidget(kpi_label)
        content_layout.addWidget(list_label)
        return build_card(title, content, role="soft-card")[0]

    def _build_kpi_card(self, title: str, kpi_label: QtWidgets.QLabel) -> QtWidgets.QFrame:
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        content_layout.addWidget(kpi_label)
        return build_card(title, content, role="soft-card")[0]

    def _build_list_card(self, title: str, list_label: QtWidgets.QLabel) -> QtWidgets.QFrame:
        content = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        content_layout.addWidget(list_label)
        return build_card(title, content, role="soft-card")[0]

    def _configure_text_box(
        self,
        widget: QtWidgets.QPlainTextEdit,
        min_lines: int = 2,
        auto_fit: bool = True,
        expand: bool = False,
    ) -> None:
        widget.setLineWrapMode(QtWidgets.QPlainTextEdit.WidgetWidth)
        widget.setWordWrapMode(QtGui.QTextOption.WordWrap)
        widget.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff if auto_fit else QtCore.Qt.ScrollBarAsNeeded
        )
        widget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        if expand:
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
            )
        else:
            widget.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
            )
        if auto_fit:
            self._text_boxes.append((widget, min_lines))

    def _fit_text_boxes(self) -> None:
        for widget, min_lines in self._text_boxes:
            width = widget.viewport().width() or widget.width() or 320
            document = widget.document()
            document.setTextWidth(width)
            height = document.size().height()
            line_height = QtGui.QFontMetrics(widget.font()).lineSpacing()
            min_height = (line_height * min_lines) + 8
            widget.setFixedHeight(int(max(height + 6, min_height)))

    def _set_overlay_visible(self, overlay: QtWidgets.QWidget, show: bool) -> None:
        if show:
            overlay.show()
            overlay.raise_()
        else:
            overlay.hide()

    def _animate_page_in(self, page: QtWidgets.QWidget) -> None:
        return

    def _connect_signals(self) -> None:
        signals.set_ui_state.connect(self.set_ui_state)
        signals.set_menu_context.connect(self.set_menu_context)
        signals.draw_ar_object.connect(self.draw_ar_object)
        signals.remove_ar_object.connect(self.remove_ar_object)
        signals.rotate_right.connect(self.rotate_right)
        signals.rotate_left.connect(self.rotate_left)
        signals.handle_customer_selection.connect(self.handle_customer_selection)
        signals.handle_admin_selection.connect(self.handle_admin_selection)
        signals.handle_manage_users_selection.connect(self.handle_manage_users_selection)
        signals.handle_edit_stock_selection.connect(self.handle_edit_stock_selection)
        signals.handle_purchases_selection.connect(self.handle_purchases_selection)
        signals.handle_inventory_selection.connect(self.handle_inventory_selection)
        signals.handle_reports_selection.connect(self.handle_reports_selection)
        signals.show_payment_state.connect(self.show_payment_state)
        signals.update_face_status.connect(self.update_face_status)
        signals.update_auth_error.connect(self.update_auth_error)
        signals.update_customer_display.connect(self.update_customer_display)
        signals.update_admin_display.connect(self.update_admin_display)
        signals.camera_frame.connect(self.update_camera_frame)
        signals.capture_calibration.connect(self.capture_gaze_calibration)

    def update_face_status(self, text: str) -> None:
        self.deepface_status.setText(text)

    def update_auth_error(self, text: str) -> None:
        self.auth_error.setText(text)
        self.auth_error.setVisible(bool(text.strip()))

    def _sync_menu_overlays(self) -> None:
        show_customer = customer_menu_open and self.state_stack.currentWidget() == self.customer_page
        show_admin = admin_menu_open and self.state_stack.currentWidget() == self.admin_page
        self._set_overlay_visible(self.customer_menu_overlay, show_customer)
        self._set_overlay_visible(self.admin_menu_overlay, show_admin)

    def set_ui_state(self, state: str) -> None:
        global customer_menu_open, admin_menu_open, manage_users_open, edit_stock_open, purchases_open, inventory_open, current_ui_state
        global customer_menu_locked, admin_menu_locked, reports_open, customer_history_active
        state_map = {
            "idle": (self.idle_page, "STATE: IDLE"),
            "customer": (self.customer_page, "STATE: CUSTOMER"),
            "admin": (self.admin_page, "STATE: ADMIN"),
            "payment": (self.payment_page, "STATE: PAYMENT"),
            "manage_users": (self.manage_users_page, "STATE: MANAGE USERS"),
            "edit_stock": (self.edit_stock_page, "STATE: EDIT STOCK"),
            "purchases": (self.purchases_page, "STATE: PURCHASES"),
            "inventory": (self.inventory_page, "STATE: INVENTORY"),
            "reports": (self.reports_page, "STATE: REPORTS"),
        }
        if state not in state_map:
            return

        page, badge = state_map[state]
        current_ui_state = state
        if self.state_stack.currentWidget() is not page:
            self.state_stack.setCurrentWidget(page)
            self._animate_page_in(page)
        self.state_badge.setText(badge)
        if state != "customer":
            customer_history_active = False
        if state == "customer":
            customer_menu_open = True
            admin_menu_open = False
            manage_users_open = False
            edit_stock_open = False
            purchases_open = False
            inventory_open = False
            reports_open = False
            self.customer_menu_widget.setFocus(QtCore.Qt.OtherFocusReason)
            customer_name = current_identity.get("name")
            customer_role = current_identity.get("role")
            if customer_role == "customer" and customer_name and customer_name != "Recognizing user":
                global calibration_active, gaze_calibrated, gaze_feature_map, gaze_affine
                loaded_map = load_gaze_calibration(customer_name)
                if loaded_map is not None:
                    gaze_feature_map = loaded_map
                    gaze_affine = None
                    gaze_calibrated = True
                    calibration_active = False
                    if hasattr(self, "calibration_overlay"):
                        self.calibration_overlay.setVisible(False)
                        self.calibration_overlay.set_target(None, None, "")
                else:
                    gaze_feature_map = None
                    gaze_affine = None
                    gaze_calibrated = False
                    if not calibration_active:
                        QtCore.QTimer.singleShot(500, self.start_gaze_calibration)
        elif state == "admin":
            customer_menu_open = False
            admin_menu_open = True
            manage_users_open = False
            edit_stock_open = False
            purchases_open = False
            inventory_open = False
            reports_open = False
            self.admin_menu_widget.setFocus(QtCore.Qt.OtherFocusReason)
        elif state == "manage_users":
            customer_menu_open = False
            admin_menu_open = False
            manage_users_open = True
            edit_stock_open = False
            purchases_open = False
            inventory_open = False
            reports_open = False
            self.manage_users_menu.setFocus(QtCore.Qt.OtherFocusReason)
        elif state == "edit_stock":
            customer_menu_open = False
            admin_menu_open = False
            manage_users_open = False
            edit_stock_open = True
            purchases_open = False
            inventory_open = False
            reports_open = False
            self.edit_stock_menu.setFocus(QtCore.Qt.OtherFocusReason)
        elif state == "purchases":
            customer_menu_open = False
            admin_menu_open = False
            manage_users_open = False
            edit_stock_open = False
            purchases_open = True
            inventory_open = False
            reports_open = False
            self.purchases_menu.setFocus(QtCore.Qt.OtherFocusReason)
        elif state == "inventory":
            customer_menu_open = False
            admin_menu_open = False
            manage_users_open = False
            edit_stock_open = False
            purchases_open = False
            inventory_open = True
            reports_open = False
            self.inventory_menu.setFocus(QtCore.Qt.OtherFocusReason)
        elif state == "reports":
            customer_menu_open = False
            admin_menu_open = False
            manage_users_open = False
            edit_stock_open = False
            purchases_open = False
            inventory_open = False
            reports_open = True
            self.reports_menu.setFocus(QtCore.Qt.OtherFocusReason)
        else:
            customer_menu_open = False
            admin_menu_open = False
            manage_users_open = False
            edit_stock_open = False
            purchases_open = False
            inventory_open = False
            reports_open = False
            customer_menu_locked = False
            admin_menu_locked = False
        if state not in ("manage_users", "edit_stock", "purchases", "inventory", "reports"):
            self._exit_manage_users_mode()
        if state != "edit_stock":
            self._exit_edit_stock_mode()
        if state != "purchases":
            self._exit_purchases_mode()
        if state != "inventory":
            self._exit_inventory_mode()
        self._sync_menu_overlays()

    def show_payment_state(self, message: str, confirmed: bool) -> None:
        if current_ui_state == "customer":
            global customer_history_active
            customer_history_active = False
            self.update_customer_display(message)
            return
        self.payment_status.setText(message)
        self.set_ui_state("payment")
        if confirmed:
            QtCore.QTimer.singleShot(1600, lambda: self.set_ui_state("customer"))

    def _exit_manage_users_mode(self) -> None:
        global manage_users_open, manage_users_mode, manage_target_user
        manage_users_open = False
        manage_users_mode = "users"
        manage_target_user = None

    def _exit_edit_stock_mode(self) -> None:
        global edit_stock_open, edit_stock_mode, edit_stock_target, selected_edit_index, edit_stock_quantity
        edit_stock_open = False
        edit_stock_mode = "items"
        edit_stock_target = None
        selected_edit_index = 0
        edit_stock_quantity = 0

    def _exit_purchases_mode(self) -> None:
        global purchases_open, purchases_mode, purchases_target_customer, selected_purchase_index
        purchases_open = False
        purchases_mode = "customers"
        purchases_target_customer = None
        selected_purchase_index = 0

    def _exit_inventory_mode(self) -> None:
        global inventory_open, inventory_mode, inventory_target_item, selected_inventory_index
        inventory_open = False
        inventory_mode = "items"
        inventory_target_item = None
        selected_inventory_index = 0

    def update_cart_display(self) -> None:
        self.cart_text.clear()
        if not cart:
            self.cart_text.setPlainText("Cart is empty.\n")
            self._fit_text_boxes()
            self.refresh_customer_metrics()
            return

        lines = []
        for pid, data in cart.items():
            name = data.get("name", "Unknown")
            qty = data.get("qty", 0)
            price = float(data.get("price", 0.0))
            line_total = price * int(qty)
            lines.append(
                f"- {name} | Qty: {qty} | Price: {format_price(price)} | Total: {format_price(line_total)}"
            )
        self.cart_text.setPlainText("\n".join(lines))
        self._fit_text_boxes()
        self.refresh_customer_metrics()

    def _compute_calibration_targets(self) -> list[tuple[str, tuple[float, float]]]:
        central = self.centralWidget()
        if central is None:
            return []
        central_size = central.size()
        if central_size.width() == 0 or central_size.height() == 0:
            return []

        edge = 0.06
        far = 1.0 - edge
        targets = [
            ("Center", (0.5, 0.5)),
            ("Top Left", (edge, edge)),
            ("Top", (0.5, edge)),
            ("Top Right", (far, edge)),
            ("Left", (edge, 0.5)),
            ("Right", (far, 0.5)),
            ("Bottom Left", (edge, far)),
            ("Bottom", (0.5, far)),
            ("Bottom Right", (far, far)),
        ]
        results = []
        for label, coords in targets:
            norm_x, norm_y = coords
            results.append((label, (float(norm_x), float(norm_y))))
        return results

    def start_gaze_calibration(self) -> None:
        global calibration_active, calibration_targets, calibration_index
        global gaze_samples, gaze_affine, gaze_calibrated, gaze_feature_map
        global calibration_buffer, calibration_points, calibration_target_started
        global calibration_capture_until, calibration_capture_samples
        global calibration_auto, calibration_auto_index, calibration_auto_next
        if calibration_active or current_ui_state != "customer":
            return
        calibration_targets = self._compute_calibration_targets()
        if not calibration_targets:
            return
        calibration_active = True
        calibration_index = 0
        calibration_auto_index = 0
        calibration_auto_next = 0.0
        gaze_samples = []
        gaze_affine = None
        gaze_feature_map = None
        gaze_calibrated = False
        calibration_buffer = []
        calibration_points = []
        calibration_target_started = None
        calibration_capture_until = 0.0
        calibration_capture_samples = []
        label, _ = calibration_targets[calibration_index]
        self.update_customer_display(
            f"Gaze calibration: follow the dot ({label})."
        )

    def capture_gaze_calibration(self) -> None:
        global calibration_active, calibration_targets, calibration_index
        global calibration_capture_until, calibration_capture_samples
        if not calibration_active:
            return
        if not calibration_targets or calibration_index >= len(calibration_targets):
            calibration_active = False
            return
        if calibration_capture_until > 0:
            return
        now = time.time()
        calibration_capture_until = now + CALIB_CAPTURE_SEC
        calibration_capture_samples = []
        label, _ = calibration_targets[calibration_index]
        self.update_customer_display(f"Hold gaze on {label}...")

    def _solve_gaze_affine(self, samples: list[tuple[tuple[float, float], tuple[float, float]]]) -> Optional[np.ndarray]:
        if len(samples) < 3:
            return None
        src = []
        dst = []
       




       
        try:
            coeff, _, _, _ = np.linalg.lstsq(src_m, dst_m, rcond=None)
        except Exception:
            return None
        return coeff.T

    def _solve_gaze_feature_map(
        self,
        samples: list[tuple[tuple[float, float, float, float, float, float], tuple[float, float]]],
    ) -> Optional[np.ndarray]:
        if len(samples) < 7:
            return None
        src = []
        dst_x = []
        dst_y = []
        for features, (sx, sy) in samples:
            if len(features) != 6:
                continue
            src.append([features[0], features[1], features[2], features[3], features[4], features[5], 1.0])
            dst_x.append(sx)
            dst_y.append(sy)
        if len(src) < 7:
            return None
        src_m = np.array(src, dtype=np.float32)
        dst_x_m = np.array(dst_x, dtype=np.float32)
        dst_y_m = np.array(dst_y, dtype=np.float32)
        try:
            ridge = 0.01
            xtx = src_m.T @ src_m
            xtx += ridge * np.eye(xtx.shape[0], dtype=xtx.dtype)
            coeff_x = np.linalg.solve(xtx, src_m.T @ dst_x_m)
            coeff_y = np.linalg.solve(xtx, src_m.T @ dst_y_m)
        except Exception:
            return None
        return np.vstack([coeff_x, coeff_y])

    def _solve_gaze_poly(self, samples: list[tuple[tuple[float, float], tuple[float, float]]]) -> Optional[np.ndarray]:
        if len(samples) < 6:
            return None
        src = []
        dst_x = []
        dst_y = []
        for (gx, gy), (sx, sy) in samples:
            src.append([gx, gy, gx * gx, gy * gy, gx * gy, 1.0])
            dst_x.append(sx)
            dst_y.append(sy)
        src_m = np.array(src, dtype=np.float32)
        dst_x_m = np.array(dst_x, dtype=np.float32)
        dst_y_m = np.array(dst_y, dtype=np.float32)
        try:
            coeff_x, _, _, _ = np.linalg.lstsq(src_m, dst_x_m, rcond=None)
            coeff_y, _, _, _ = np.linalg.lstsq(src_m, dst_y_m, rcond=None)
        except Exception:
            return None
        return np.vstack([coeff_x, coeff_y])

    def update_gaze_scroll(self) -> None:
        if current_ui_state != "customer" or not customer_history_active:
            return
        if latest_frame is None:
            return
        now = time.time()
        gaze_value = gaze_scroll_value
        stamp = gaze_scroll_timestamp
        if gaze_value is None or (now - stamp) > 0.6:
            return #**
        direction = None
        if gaze_value <= GAZE_SCROLL_TOP:
            direction = "up"
        elif gaze_value >= GAZE_SCROLL_BOTTOM:
            direction = "down"

        if direction is None:
            self.gaze_scroll_dir = None
            self.gaze_scroll_start = 0.0
            return

        if self.gaze_scroll_dir != direction:
            self.gaze_scroll_dir = direction
            self.gaze_scroll_start = now
            return

        if (now - self.gaze_scroll_start) < GAZE_SCROLL_DWELL:
            return
        if (now - self.gaze_scroll_last) < GAZE_SCROLL_INTERVAL:
            return

        scrollbar = self.customer_display.verticalScrollBar()
        if scrollbar.maximum() <= 0:
            return

        delta = -GAZE_SCROLL_STEP if direction == "up" else GAZE_SCROLL_STEP
        next_value = max(0, min(scrollbar.maximum(), scrollbar.value() + delta))
        scrollbar.setValue(next_value)
        self.gaze_scroll_last = now

    def update_gaze_overlay(self) -> None:
        if not hasattr(self, "gaze_overlay"):
            return
        if not GAZE_SHOW_DOT:
            self.gaze_overlay.setVisible(False)
            self.gaze_overlay.set_position(None, None)
            return
        if current_ui_state != "customer":
            self.gaze_overlay.setVisible(False)
            self.gaze_overlay.set_position(None, None)
            return
        self._update_gaze_overlay_geometry()
        now = time.time()
        if gaze_point_mapped is not None and (now - gaze_point_mapped_timestamp) <= 0.6:
            gx, gy = gaze_point_mapped
        else:
            if gaze_point_norm is None or (now - gaze_point_timestamp) > 0.6:
                self.gaze_overlay.setVisible(False)
                self.gaze_overlay.set_position(None, None)
                return
            gx, gy = map_gaze_point(gaze_point_norm[0], gaze_point_norm[1])
        self.gaze_overlay.setVisible(True)
        self.gaze_overlay.set_position(gx, gy)

    def update_calibration_progress(self) -> None:
        global calibration_active, calibration_targets, calibration_index
        global calibration_buffer, calibration_points, calibration_target_started
        global calibration_capture_until, calibration_capture_samples
        global gaze_features, gaze_features_timestamp, gaze_stable, gaze_stable_timestamp
        global gaze_feature_map, gaze_affine, gaze_calibrated, session_heatmap_acc
        global calibration_auto, calibration_auto_index, calibration_auto_next
        if GAZE_AUTO_MODE:
            if hasattr(self, "calibration_overlay"):
                self.calibration_overlay.setVisible(False)
                self.calibration_overlay.set_target(None, None, "")
            return
        if not calibration_active or current_ui_state != "customer":
            if hasattr(self, "calibration_overlay"):
                self.calibration_overlay.setVisible(False)
                self.calibration_overlay.set_target(None, None, "")
            return

        if not calibration_targets or calibration_index >= len(calibration_targets):
            return

        now = time.time()
        if calibration_auto:
            if calibration_auto_next == 0.0:
                calibration_auto_next = now
            if now >= calibration_auto_next:
                calibration_index = calibration_auto_index
                if calibration_index >= len(calibration_targets):
                    calibration_index = len(calibration_targets) - 1
                label, target = calibration_targets[calibration_index]
                calibration_capture_until = now + CALIB_CAPTURE_SEC
                calibration_capture_samples = []
                calibration_auto_next = calibration_capture_until + CALIB_DWELL_SEC
                if hasattr(self, "calibration_overlay"):
                    self.calibration_overlay.setVisible(True)
                    self.calibration_overlay.set_target(target[0], target[1], label)
                self.update_customer_display(f"Gaze calibration: follow the dot ({label}).")
            else:
                label, target = calibration_targets[calibration_index]
                if hasattr(self, "calibration_overlay"):
                    self.calibration_overlay.setVisible(True)
                    self.calibration_overlay.set_target(target[0], target[1], label)
        else:
            label, target = calibration_targets[calibration_index]
            if hasattr(self, "calibration_overlay"):
                self.calibration_overlay.setVisible(True)
                self.calibration_overlay.set_target(target[0], target[1], label)

        if calibration_capture_until <= 0:
            return

        if (
            not gaze_stable
            or (now - gaze_stable_timestamp) > CALIB_STABLE_TIMEOUT
            or gaze_features is None
            or (now - gaze_features_timestamp) > CALIB_STABLE_TIMEOUT
        ):
            return

        calibration_capture_samples.append(gaze_features)
        if now < calibration_capture_until:
            return

        if len(calibration_capture_samples) < CALIB_MIN_SAMPLES:
            calibration_capture_until = 0.0
            calibration_capture_samples = []
            if not calibration_auto:
                self.update_customer_display("Calibration point failed. Tap marker 9 to retry.")
            return

        features_med = np.median(np.array(calibration_capture_samples), axis=0)
        calibration_points.append((tuple(features_med.tolist()), target))
        calibration_index += 1
        if calibration_auto:
            calibration_auto_index += 1
        calibration_capture_until = 0.0
        calibration_capture_samples = []

        if calibration_index >= len(calibration_targets):
            gaze_feature_map = self._solve_gaze_feature_map(calibration_points)
            gaze_affine = None
            calibration_active = False
            gaze_calibrated = gaze_feature_map is not None
            with session_lock:
                session_heatmap_acc = None
            if hasattr(self, "calibration_overlay"):
                self.calibration_overlay.setVisible(False)
                self.calibration_overlay.set_target(None, None, "")
            if gaze_calibrated:
                name = current_identity.get("name")
                role = current_identity.get("role")
                if role == "customer" and name and name != "Recognizing user":
                    save_gaze_calibration(name, gaze_feature_map)
                self.update_customer_display("Calibration complete. You can continue.")
            else:
                self.update_customer_display("Calibration failed. Try again.")
            return

        if not calibration_auto:
            next_label, _ = calibration_targets[calibration_index]
            self.update_customer_display(
                f"Gaze calibration: look at {next_label} and tap marker 9."
            )

    def scroll_customer_history(self, direction: str) -> None:
        if current_ui_state != "customer" or not customer_history_active:
            return
        scrollbar = self.customer_display.verticalScrollBar()
        if scrollbar.maximum() <= 0:
            return
        delta = -MARKER_SCROLL_STEP if direction == "up" else MARKER_SCROLL_STEP
        next_value = max(0, min(scrollbar.maximum(), scrollbar.value() + delta))
        scrollbar.setValue(next_value)

    def refresh_customer_metrics(self) -> None:
        total_items = sum(int(item.get("qty", 0)) for item in cart.values())
        subtotal = sum(
            float(item.get("price", 0.0)) * int(item.get("qty", 0))
            for item in cart.values()
        )
        with face_status_lock:
            active_name = current_identity["name"]
        with last_action_lock:
            last_action = last_action_message or "--"

        self.customer_cart_kpi.setText(f"{total_items} Items")
        self.customer_total_kpi.setText(format_price(subtotal))
        self.customer_summary.setText(
            f"Subtotal: {format_price(subtotal)}\nItems: {total_items}\nLast action: {last_action}"
        )

        self.customer_badges[0].setText(
            "Session Active" if session_active else "Session Idle"
        )
        self.customer_badges[1].setText(
            "Cart Ready" if total_items > 0 else "Cart Empty"
        )
        with socket_clients_lock:
            tuio_ok = socket_clients > 0
        self.customer_badges[2].setText("TUIO Connected" if tuio_ok else "TUIO Offline")
        self.customer_badges[3].setText(
            "Heatmaps Active" if session_active else "Heatmaps Idle"
        )

    def set_menu_context(self, menu_name: str) -> None:
        global customer_menu_open, admin_menu_open, manage_users_open, edit_stock_open, purchases_open, inventory_open
        global selected_customer_index, selected_admin_index, manage_users_mode, manage_target_user
        global customer_menu_locked, admin_menu_locked
        with face_status_lock:
            role = current_identity["role"]
        if menu_name == "customer" and role != "customer":
            log_event(f"Ignored customer menu request while role is {role}.")
            return
        if menu_name == "admin" and role != "admin":
            log_event(f"Ignored admin menu request while role is {role}.")
            return
        if menu_name == "customer":
            customer_menu_open, admin_menu_open, manage_users_open = True, False, False
            edit_stock_open = False
            purchases_open = False
            inventory_open = False
            customer_menu_locked = False
            selected_customer_index = 0
            self.set_ui_state("customer")
            self.customer_menu_title.setText("Customer Menu")
            self.customer_menu_widget.set_options(customer_menu_options)
            self.customer_menu_widget.set_selected_index(selected_customer_index)
            self._sync_menu_overlays()
            log_event("Switched to customer circular menu.")
        elif menu_name == "admin":
            customer_menu_open, admin_menu_open, manage_users_open = False, True, False
            edit_stock_open = False
            purchases_open = False
            inventory_open = False
            admin_menu_locked = False
            manage_users_mode = "users"
            manage_target_user = None
            selected_admin_index = 0
            self.set_ui_state("admin")
            self.admin_menu_title.setText("Admin Menu")
            self.admin_menu_widget.set_options(admin_menu_options)
            self.admin_menu_widget.set_selected_index(selected_admin_index)
            self._sync_menu_overlays()
            log_event("Switched to admin circular menu.")

    def update_customer_display(self, text: str) -> None:
        self.customer_display.setPlainText(text)
        self._fit_text_boxes()

    def update_admin_display(self, text: str) -> None:
        self.admin_display.setPlainText(text)
        self._fit_text_boxes()

    def refresh_admin_metrics(self) -> None:
        items = load_inventory_items()
        total_items = len(items)
        total_units = sum(item.get("stock", 0) for item in items)
        low_items = [item for item in items if item.get("stock", 0) <= LOW_STOCK_THRESHOLD]
        low_names = ", ".join(item["name"] for item in low_items[:3]) or "--"

        self.inventory_kpi.setText(f"{total_items} Items")
        self.inventory_list.setText(
            f"Low stock: {len(low_items)}\nTotal units: {total_units}"
        )

        sales = load_sales_metrics()
        total_orders = sales["total_orders"]
        total_sold = sales["total_qty"]
        latest_purchase = sales["latest"]
        revenue_total = sales["total_price"]
        self.sales_kpi.setText(f"{total_orders} Orders")
        self.sales_list.setText(
            f"Items sold: {total_sold}\nLatest: {latest_purchase}\nRevenue: {format_price(revenue_total)}"
        )

        with face_status_lock:
            active_name = current_identity["name"]
        heatmap_count = 0
        with db_session() as conn:
            row = conn.execute("SELECT COUNT(*) FROM heatmaps").fetchone()
            if row:
                heatmap_count = int(row[0])
        self.activity_kpi.setText(active_name or "No Users")
        with last_action_lock:
            last_action = last_action_message or "--"
        self.activity_list.setText(
            f"State: {current_ui_state}\nHeatmaps: {heatmap_count}\nLast action: {last_action}"
        )

        camera_ok = latest_frame is not None
        with socket_clients_lock:
            tuio_ok = socket_clients > 0
        heatmap_ok = session_active

        self.status_badges[0].setText("System Online")
        self.status_badges[1].setText("Camera OK" if camera_ok else "Camera Offline")
        self.status_badges[2].setText("TUIO Connected" if tuio_ok else "TUIO Offline")
        self.status_badges[3].setText("Heatmaps Active" if heatmap_ok else "Heatmaps Idle")

    def _lock_customer_menu(self, action: str) -> None:
        global customer_menu_locked, customer_menu_last_index
        if customer_menu_locked:
            return
        customer_menu_last_index = selected_customer_index
        customer_menu_locked = True
        self.customer_menu_title.setText(f"Customer Menu: {action}")
        self.customer_menu_widget.set_options(["Back"])
        self.customer_menu_widget.set_selected_index(0)

    def _unlock_customer_menu(self) -> None:
        global customer_menu_locked, selected_customer_index
        if not customer_menu_locked:
            return
        customer_menu_locked = False
        selected_customer_index = customer_menu_last_index
        self.customer_menu_title.setText("Customer Menu")
        self.customer_menu_widget.set_options(customer_menu_options)
        self.customer_menu_widget.set_selected_index(selected_customer_index)

    def _lock_admin_menu(self, action: str) -> None:
        global admin_menu_locked, admin_menu_last_index
        if admin_menu_locked:
            return
        admin_menu_last_index = selected_admin_index
        admin_menu_locked = True
        self.admin_menu_title.setText(f"Admin Menu: {action}")
        self.admin_menu_widget.set_options(["Back"])
        self.admin_menu_widget.set_selected_index(0)

    def _unlock_admin_menu(self) -> None:
        global admin_menu_locked, selected_admin_index
        if not admin_menu_locked:
            return
        admin_menu_locked = False
        selected_admin_index = admin_menu_last_index
        self.admin_menu_title.setText("Admin Menu")
        self.admin_menu_widget.set_options(admin_menu_options)
        self.admin_menu_widget.set_selected_index(selected_admin_index)

    def refresh_manage_users_menu(self) -> None:
        options = []
        with db_session() as conn:
            cursor = conn.execute("SELECT name, role FROM faces ORDER BY name")
            for name, role in cursor.fetchall():
                options.append(f"{name} ({role})")

        if not options:
            options = ["No Users"]

        options.append("Back to Admin")

        self.manage_users_menu_title.setText("Users")
        self.manage_users_menu.set_options(options)
        self.manage_users_menu.set_selected_index(0)
        self.manage_users_status.setText("Select a user.")
        self._manage_users_options = options

    def refresh_edit_stock_menu(self) -> None:
        global edit_stock_mode, edit_stock_target, selected_edit_index
        items = load_inventory_items()
        options = [item["name"] for item in items]
        if not options:
            options = ["No Inventory"]
        options.append("Back to Admin")
        self.edit_stock_menu_title.setText("Medicines")
        self.edit_stock_menu.set_options(options)
        self.edit_stock_menu.set_selected_index(0)
        self.edit_stock_status.setText("Select a medicine.")
        self._edit_stock_items = items
        edit_stock_mode = "items"
        edit_stock_target = None
        selected_edit_index = 0

    def refresh_purchases_menu(self) -> None:
        global purchases_mode, purchases_target_customer, selected_purchase_index
        options = load_purchase_customers()
        if not options:
            options = ["No Purchases"]
        options.append("Back to Admin")
        self.purchases_menu_title.setText("Customers")
        self.purchases_menu.set_options(options)
        self.purchases_menu.set_selected_index(0)
        self.purchases_status.setText("Select a customer.")
        purchases_mode = "customers"
        purchases_target_customer = None
        selected_purchase_index = 0

    def refresh_inventory_menu(self) -> None:
        global inventory_mode, inventory_target_item, selected_inventory_index
        items = load_inventory_items()
        options = [item["name"] for item in items]
        if not options:
            options = ["No Inventory"]
        options.append("Back to Admin")
        self.inventory_menu_title.setText("Medicines")
        self.inventory_menu.set_options(options)
        self.inventory_menu.set_selected_index(0)
        self.inventory_status.setText("Select a medicine.")
        self._inventory_items = items
        inventory_mode = "items"
        inventory_target_item = None
        selected_inventory_index = 0

    def set_edit_stock_action_menu(self, item: dict) -> None:
        name = item.get("name", "Unknown")
        stock = item.get("stock", 0)
        self.edit_stock_menu_title.setText("Actions")
        self.edit_stock_menu.set_options(["Update Quantity", "Delete", "Back"])
        self.edit_stock_menu.set_selected_index(0)
        self.edit_stock_status.setText(f"{name} | Stock: {stock}")

    def set_edit_stock_quantity_menu(self, item: dict) -> None:
        name = item.get("name", "Unknown")
        stock = item.get("stock", 0)
        self.edit_stock_menu_title.setText("Quantity")
        self.edit_stock_menu.set_options(["Increase", "Decrease", "Confirm", "Back"])
        self.edit_stock_menu.set_selected_index(0)
        self.edit_stock_status.setText(f"{name} | Qty: {stock}")

    def update_inventory_stock(self, name: str, new_stock: int) -> None:
        with db_session() as conn:
            conn.execute(
                "UPDATE inventory SET stock = ?, updated_at = ? WHERE name = ?",
                (int(new_stock), time.strftime("%Y-%m-%d %H:%M:%S"), name),
            )
        export_database_sql(DATABASE_SQL_PATH)

    def delete_inventory_item(self, name: str) -> None:
        with db_session() as conn:
            conn.execute("DELETE FROM inventory WHERE name = ?", (name,))
        export_database_sql(DATABASE_SQL_PATH)

    def set_manage_users_action_menu(self, user_label: str) -> None:
        self.manage_users_menu_title.setText("Actions")
        self.manage_users_menu.set_options(["Set Admin", "Set Customer", "Delete User", "Back"])
        self.manage_users_menu.set_selected_index(0)
        self.manage_users_status.setText(f"Selected: {user_label}")

    def apply_user_role_change(self, user_label: str, new_role: str) -> None:
        if "(" not in user_label:
            return
        name = user_label.split("(", 1)[0].strip()
        with db_session() as conn:
            conn.execute(
                "UPDATE faces SET role = ? WHERE name = ?",
                (new_role, name),
            )
        face_rec.load_known_users()
        export_database_sql(DATABASE_SQL_PATH)
        self.manage_users_status.setText(f"Updated {name} -> {new_role}")
        log_event(f"Updated role for {name} to {new_role}.")

    def delete_user_record(self, user_label: str) -> None:
        if "(" not in user_label:
            return
        name = user_label.split("(", 1)[0].strip()
        with db_session() as conn:
            conn.execute("DELETE FROM faces WHERE name = ?", (name,))
        export_database_sql(DATABASE_SQL_PATH)
        face_rec.load_known_users()
        self.manage_users_status.setText(f"Deleted user {name}. Records kept.")
        log_event(f"Deleted user {name}.")

    def handle_manage_users_selection(self) -> None:
        global manage_users_mode, manage_target_user

        if manage_users_mode == "users":
            if not getattr(self, "_manage_users_options", None):
                return
            selected = self._manage_users_options[self.manage_users_menu.selected_index]
            if selected == "Back to Admin":
                self.set_menu_context("admin")
                return
            if selected == "No Users":
                self.manage_users_status.setText("No enrolled users found.")
                return
            manage_target_user = selected
            manage_users_mode = "actions"
            self.set_manage_users_action_menu(selected)
            return

        if manage_users_mode == "actions":
            option = self.manage_users_menu.options[self.manage_users_menu.selected_index]
            if option == "Back":
                manage_users_mode = "users"
                manage_target_user = None
                self.refresh_manage_users_menu()
                return
            if manage_target_user is None:
                self.manage_users_status.setText("No user selected.")
                return
            if option == "Set Admin":
                self.apply_user_role_change(manage_target_user, "admin")
            elif option == "Set Customer":
                self.apply_user_role_change(manage_target_user, "customer")
            elif option == "Delete User":
                self.delete_user_record(manage_target_user)
                manage_users_mode = "users"
                manage_target_user = None
                self.refresh_manage_users_menu()

    def handle_edit_stock_selection(self) -> None:
        global edit_stock_mode, edit_stock_target, edit_stock_quantity

        if edit_stock_mode == "items":
            options = getattr(self, "_edit_stock_items", [])
            choice = self.edit_stock_menu.options[self.edit_stock_menu.selected_index]
            if choice == "Back to Admin":
                self.set_menu_context("admin")
                return
            if choice == "No Inventory":
                self.edit_stock_status.setText("No inventory items found.")
                return
            target = next((item for item in options if item["name"] == choice), None)
            if not target:
                self.edit_stock_status.setText("Invalid selection.")
                return
            edit_stock_target = target
            edit_stock_mode = "actions"
            self.set_edit_stock_action_menu(target)
            return

        if edit_stock_mode == "actions":
            if not edit_stock_target:
                self.edit_stock_status.setText("No medicine selected.")
                return
            option = self.edit_stock_menu.options[self.edit_stock_menu.selected_index]
            name = edit_stock_target.get("name", "Unknown")
            if option == "Back":
                self.refresh_edit_stock_menu()
                return
            if option == "Delete":
                self.delete_inventory_item(name)
                self.edit_stock_status.setText(f"Deleted {name}.")
                self.refresh_edit_stock_menu()
                return
            if option == "Update Quantity":
                edit_stock_quantity = int(edit_stock_target.get("stock", 0))
                edit_stock_mode = "quantity"
                self.set_edit_stock_quantity_menu(edit_stock_target)
                return

        if edit_stock_mode == "quantity":
            if not edit_stock_target:
                self.edit_stock_status.setText("No medicine selected.")
                return
            option = self.edit_stock_menu.options[self.edit_stock_menu.selected_index]
            name = edit_stock_target.get("name", "Unknown")
            if option == "Increase":
                edit_stock_quantity += 1
                self.edit_stock_status.setText(f"{name} | Qty: {edit_stock_quantity}")
                return
            if option == "Decrease":
                edit_stock_quantity = max(0, edit_stock_quantity - 1)
                self.edit_stock_status.setText(f"{name} | Qty: {edit_stock_quantity}")
                return
            if option == "Confirm":
                self.update_inventory_stock(name, edit_stock_quantity)
                edit_stock_target["stock"] = edit_stock_quantity
                self.edit_stock_status.setText(f"Updated {name} -> {edit_stock_quantity}")
                edit_stock_mode = "actions"
                self.set_edit_stock_action_menu(edit_stock_target)
                return
            if option == "Back":
                edit_stock_mode = "actions"
                self.set_edit_stock_action_menu(edit_stock_target)
                return

    def handle_purchases_selection(self) -> None:
        global purchases_mode, purchases_target_customer
        if purchases_mode != "customers":
            purchases_mode = "customers"

        choice = self.purchases_menu.options[self.purchases_menu.selected_index]
        if choice == "Back to Admin":
            self.set_menu_context("admin")
            return
        if choice == "No Purchases":
            self.purchases_status.setText("No purchase records found.")
            self.purchases_text.setPlainText("No purchase records found.")
            return

        purchases_target_customer = choice
        history = load_purchase_history(choice, limit=15)
        if not history:
            self.purchases_text.setPlainText("No purchases found for this customer.")
        else:
            lines = [f"Purchase History: {choice}"]
            for entry in history:
                entry_total = float(entry.get("total_price", 0.0))
                if entry_total <= 0:
                    entry_total = calculate_items_total(entry.get("items", []))
                lines.append(
                    f"- {entry['created_at']} | Qty: {entry['total_qty']} | Total: {format_price(entry_total)}"
                )
                for item in entry["items"]:
                    item_name = item.get("name", "Unknown")
                    item_qty = item.get("qty", 0)
                    item_price = float(item.get("price", 0.0))
                    line_total = item_price * item_qty
                    lines.append(
                        f"  - {item_name} x{item_qty} @ {format_price(item_price)} = {format_price(line_total)}"
                    )
            self.purchases_text.setPlainText("\n".join(lines))
        self.purchases_status.setText(f"Selected: {choice}")

    def handle_inventory_selection(self) -> None:
        global inventory_mode, inventory_target_item
        if inventory_mode != "items":
            inventory_mode = "items"

        choice = self.inventory_menu.options[self.inventory_menu.selected_index]
        if choice == "Back to Admin":
            self.set_menu_context("admin")
            return
        if choice == "No Inventory":
            self.inventory_status.setText("No inventory items found.")
            self.inventory_text.setPlainText("No inventory items found.")
            return

        item = next((entry for entry in getattr(self, "_inventory_items", []) if entry["name"] == choice), None)
        if not item:
            self.inventory_status.setText("Invalid selection.")
            return
        inventory_target_item = item
        marker = item.get("marker_id")
        marker_text = "-" if marker is None else str(marker)
        lines = [
            f"Medicine: {item['name']}",
            f"Stock: {item['stock']}",
            f"Marker: {marker_text}",
            f"Price: {format_price(float(item['price']))}",
        ]
        self.inventory_text.setPlainText("\n".join(lines))
        self.inventory_status.setText(f"Selected: {item['name']}")

    def enter_reports_mode(self) -> None:
        global reports_mode, reports_target_customer
        reports_mode = "customers"
        reports_target_customer = None
        options = load_customer_names()
        if not options:
            options = ["No Customers"]
        options.append("Back to Admin")
        self.reports_menu_title.setText("Customers")
        self.reports_menu.set_options(options)
        self.reports_menu.set_selected_index(0)
        self.reports_image.setText("Select a customer to view heatmap.")
        self.reports_image.setPixmap(QtGui.QPixmap())
        self.reports_caption.setText("")
        self.set_ui_state("reports")

    def _set_reports_type_menu(self, customer_name: str) -> None:
        self.reports_menu_title.setText("Heatmaps")
        self.reports_menu.set_options(["Latest", "Average", "Back"])
        self.reports_menu.set_selected_index(0)
        self.reports_caption.setText(f"Customer: {customer_name}")

    def _show_report_image(self, customer_name: str, kind: str) -> None:
        path = load_heatmap_path(customer_name, kind)
        if not path.exists():
            self.reports_image.setText(f"No {kind} heatmap found for {customer_name}.")
            self.reports_image.setPixmap(QtGui.QPixmap())
            return
        pixmap = QtGui.QPixmap(str(path))
        if pixmap.isNull():
            self.reports_image.setText(f"Unable to load {path.name}.")
            self.reports_image.setPixmap(QtGui.QPixmap())
            return
        self.reports_image.setText("")
        self.reports_image.setPixmap(
            pixmap.scaled(
                self.reports_image.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def handle_reports_selection(self) -> None:
        global reports_mode, reports_target_customer
        if not self.reports_menu.options:
            return
        chosen = self.reports_menu.options[self.reports_menu.selected_index]

        if reports_mode == "customers":
            if chosen == "Back to Admin":
                self.set_ui_state("admin")
                return
            if chosen == "No Customers":
                self.reports_caption.setText("No customer records found.")
                return
            reports_target_customer = chosen
            reports_mode = "types"
            self._set_reports_type_menu(chosen)
            return

        if reports_mode == "types":
            if chosen == "Back":
                reports_mode = "customers"
                reports_target_customer = None
                self.enter_reports_mode()
                return
            if not reports_target_customer:
                self.reports_caption.setText("No customer selected.")
                return
            kind = "latest" if chosen == "Latest" else "average"
            self._show_report_image(reports_target_customer, kind)
            self.reports_caption.setText(
                f"Customer: {reports_target_customer} | {chosen} heatmap"
            )

    def handle_customer_selection(self) -> None:
        if not self.customer_menu_widget.options:
            return
        chosen = self.customer_menu_widget.options[self.customer_menu_widget.selected_index]
        if customer_menu_locked:
            if chosen == "Back":
                self._unlock_customer_menu()
            return
        if chosen == "Exit":
            log_event("Customer requested exit.")
            finalize_customer_session()
            QtWidgets.QApplication.quit()
            return
        if chosen == "Pay Now":
            global customer_history_active
            customer_history_active = False
            self.update_customer_display("Payment requested.\nShow Payment Marker (11) to confirm.")
            self.show_payment_state("Waiting for payment marker (11).", False)
            log_event("Pay Now selected. Waiting for payment marker (11).")
        elif chosen == "Purchase History":
            customer_history_active = True
            with face_status_lock:
                name = current_identity["name"]
                role = current_identity["role"]
            with session_lock:
                active_name = active_customer_name
            if active_name:
                name = active_name
            if role != "customer":
                self.update_customer_display("Purchase history is only available to customers.")
            else:
                history = load_purchase_history(name, limit=6)
                if not history:
                    self.update_customer_display("No purchase history found yet.")
                else:
                    lines = [f"Purchase History ({name}):"]
                    for entry in history:
                        entry_total = float(entry.get("total_price", 0.0))
                        if entry_total <= 0:
                            entry_total = calculate_items_total(entry.get("items", []))
                        lines.append(
                            f"- {entry['created_at']} | Qty: {entry['total_qty']} | Total: {format_price(entry_total)}"
                        )
                        for item in entry["items"]:
                            item_name = item.get("name", "Unknown")
                            item_qty = item.get("qty", 0)
                            item_price = float(item.get("price", 0.0))
                            line_total = item_price * item_qty
                            lines.append(
                                f"  - {item_name} x{item_qty} @ {format_price(item_price)} = {format_price(line_total)}"
                            )
                    self.update_customer_display("\n".join(lines))
                    self.customer_display.verticalScrollBar().setValue(0)
        elif chosen == "Change User":
            global customer_menu_open, admin_menu_open, manage_users_open
            customer_history_active = False
            customer_menu_open = False
            admin_menu_open = False
            manage_users_open = False
            finalize_customer_session()
            with face_status_lock:
                current_identity["name"] = "Recognizing user"
                current_identity["role"] = "unknown"
                current_identity["confidence"] = 0.0
            update_face_status_text()
            self.set_ui_state("idle")

    def handle_admin_selection(self) -> None:
        if not self.admin_menu_widget.options:
            return
        chosen = self.admin_menu_widget.options[self.admin_menu_widget.selected_index]
        if admin_menu_locked:
            if chosen == "Back":
                self._unlock_admin_menu()
            return
        if chosen == "Exit":
            log_event("Admin requested exit.")
            QtWidgets.QApplication.quit()
            return
        if chosen == "View Inventory":
            self.enter_inventory_mode()
        elif chosen == "Edit Stock":
            self.enter_edit_stock_mode()
        elif chosen == "View Purchases":
            self.enter_purchases_mode()
        elif chosen == "Reports":
            self.enter_reports_mode()
        elif chosen == "Manage Users":
            self.enter_manage_users_mode()
        elif chosen == "Enroll Customer":
            self.launch_enroll_customer()
        elif chosen == "Change User":
            global customer_menu_open, admin_menu_open, manage_users_open
            customer_menu_open = False
            admin_menu_open = False
            manage_users_open = False
            with face_status_lock:
                current_identity["name"] = "Recognizing user"
                current_identity["role"] = "unknown"
                current_identity["confidence"] = 0.0
            update_face_status_text()
            self.set_ui_state("idle")

    def launch_enroll_customer(self) -> None:
        script_path = BASE_DIR / "enroll_customer.py"
        if not script_path.exists():
            self.update_admin_display("Enroll customer script not found.")
            log_event("Enroll customer script missing.", level="error")
            return
        try:
            subprocess.Popen([sys.executable, str(script_path)], cwd=str(BASE_DIR))
            self.update_admin_display("Enroll customer window opened.")
            log_event("Opened enroll_customer.py.")
        except Exception as exc:
            self.update_admin_display("Failed to open enroll customer window.")
            log_event(f"Failed to launch enroll_customer.py: {exc}", level="error")

    def show_admin_inventory_panel(self) -> None:
        items = load_inventory_items()
        if not items:
            self.admin_inventory_text.setPlainText("No inventory items found.")
            self._fit_text_boxes()
            self.admin_inventory_card.show()
            return

        lines = []
        for item in items:
            marker = "-" if item["marker_id"] is None else str(item["marker_id"])
            lines.append(
                f"{item['name']} | Stock: {item['stock']} | Marker: {marker}"
            )
        self.admin_inventory_text.setPlainText("\n".join(lines))
        self._fit_text_boxes()
        self.admin_inventory_card.show()

    def enter_manage_users_mode(self) -> None:
        global customer_menu_open, admin_menu_open, manage_users_open
        global manage_users_mode, manage_target_user, selected_manage_index
        customer_menu_open = False
        admin_menu_open = False
        manage_users_open = True
        manage_users_mode = "users"
        manage_target_user = None
        selected_manage_index = 0
        self.refresh_manage_users_menu()
        self.set_ui_state("manage_users")

    def enter_edit_stock_mode(self) -> None:
        global customer_menu_open, admin_menu_open, manage_users_open, edit_stock_open
        global edit_stock_mode, edit_stock_target, selected_edit_index, edit_stock_quantity
        customer_menu_open = False
        admin_menu_open = False
        manage_users_open = False
        edit_stock_open = True
        edit_stock_mode = "items"
        edit_stock_target = None
        selected_edit_index = 0
        edit_stock_quantity = 0
        self.refresh_edit_stock_menu()
        self.set_ui_state("edit_stock")

    def enter_purchases_mode(self) -> None:
        global customer_menu_open, admin_menu_open, manage_users_open, edit_stock_open
        global purchases_mode, purchases_target_customer, selected_purchase_index
        customer_menu_open = False
        admin_menu_open = False
        manage_users_open = False
        edit_stock_open = False
        purchases_mode = "customers"
        purchases_target_customer = None
        selected_purchase_index = 0
        self.refresh_purchases_menu()
        self.set_ui_state("purchases")

    def enter_inventory_mode(self) -> None:
        global customer_menu_open, admin_menu_open, manage_users_open, edit_stock_open
        global inventory_mode, inventory_target_item, selected_inventory_index
        customer_menu_open = False
        admin_menu_open = False
        manage_users_open = False
        edit_stock_open = False
        inventory_mode = "items"
        inventory_target_item = None
        selected_inventory_index = 0
        self.refresh_inventory_menu()
        self.set_ui_state("inventory")

    def rotate_right(self, pid: int) -> None:
        global selected_customer_index, selected_admin_index, selected_manage_index, selected_edit_index, selected_purchase_index, selected_inventory_index
        if pid == 12:
            self.scroll_customer_history("down")
            return
        if pid in cart:
            stock_row = get_inventory_stock_by_marker(pid)
            if stock_row is not None:
                name, stock_qty = stock_row
                if cart[pid]["qty"] >= stock_qty:
                    self.update_customer_display(
                        f"Only {stock_qty} {name} left in stock."
                    )
                    log_event(f"Stock limit reached for {name}.")
                    return
            cart[pid]["qty"] += 1
            log_event(f"Qty++ {cart[pid]['name']} -> {cart[pid]['qty']}")
        elif pid == 10 and customer_menu_open and not customer_menu_locked:
            selected_customer_index = (selected_customer_index + 1) % len(customer_menu_options)
            self.customer_menu_widget.set_selected_index(selected_customer_index)
            log_event(f"Customer menu -> {customer_menu_options[selected_customer_index]}")
        elif pid == 10 and manage_users_open:
            if self.manage_users_menu.options:
                selected_manage_index = (selected_manage_index + 1) % len(self.manage_users_menu.options)
                self.manage_users_menu.set_selected_index(selected_manage_index)
        elif pid == 10 and edit_stock_open:
            if self.edit_stock_menu.options:
                selected_edit_index = (selected_edit_index + 1) % len(self.edit_stock_menu.options)
                self.edit_stock_menu.set_selected_index(selected_edit_index)
        elif pid == 10 and purchases_open:
            if self.purchases_menu.options:
                selected_purchase_index = (selected_purchase_index + 1) % len(self.purchases_menu.options)
                self.purchases_menu.set_selected_index(selected_purchase_index)
        elif pid == 10 and inventory_open:
            if self.inventory_menu.options:
                selected_inventory_index = (selected_inventory_index + 1) % len(self.inventory_menu.options)
                self.inventory_menu.set_selected_index(selected_inventory_index)
        elif pid == 10 and reports_open:
            if self.reports_menu.options:
                self.reports_menu.select_next()
        elif pid == 10 and admin_menu_open and not admin_menu_locked:
            selected_admin_index = (selected_admin_index + 1) % len(admin_menu_options)
            self.admin_menu_widget.set_selected_index(selected_admin_index)
            log_event(f"Admin menu -> {admin_menu_options[selected_admin_index]}")

    def rotate_left(self, pid: int) -> None:
        global selected_customer_index, selected_admin_index, selected_manage_index, selected_edit_index, selected_purchase_index, selected_inventory_index
        if pid == 12:
            self.scroll_customer_history("up")
            return
        if pid in cart:
            cart[pid]["qty"] = max(0, cart[pid]["qty"] - 1)
            log_event(f"Qty-- {cart[pid]['name']} -> {cart[pid]['qty']}")
        elif pid == 10 and customer_menu_open and not customer_menu_locked:
            selected_customer_index = (selected_customer_index - 1) % len(customer_menu_options)
            self.customer_menu_widget.set_selected_index(selected_customer_index)
            log_event(f"Customer menu -> {customer_menu_options[selected_customer_index]}")
        elif pid == 10 and manage_users_open:
            if self.manage_users_menu.options:
                selected_manage_index = (selected_manage_index - 1) % len(self.manage_users_menu.options)
                self.manage_users_menu.set_selected_index(selected_manage_index)
        elif pid == 10 and edit_stock_open:
            if self.edit_stock_menu.options:
                selected_edit_index = (selected_edit_index - 1) % len(self.edit_stock_menu.options)
                self.edit_stock_menu.set_selected_index(selected_edit_index)
        elif pid == 10 and purchases_open:
            if self.purchases_menu.options:
                selected_purchase_index = (selected_purchase_index - 1) % len(self.purchases_menu.options)
                self.purchases_menu.set_selected_index(selected_purchase_index)
        elif pid == 10 and inventory_open:
            if self.inventory_menu.options:
                selected_inventory_index = (selected_inventory_index - 1) % len(self.inventory_menu.options)
                self.inventory_menu.set_selected_index(selected_inventory_index)
        elif pid == 10 and reports_open:
            if self.reports_menu.options:
                self.reports_menu.select_prev()
        elif pid == 10 and admin_menu_open and not admin_menu_locked:
            selected_admin_index = (selected_admin_index - 1) % len(admin_menu_options)
            self.admin_menu_widget.set_selected_index(selected_admin_index)
            log_event(f"Admin menu -> {admin_menu_options[selected_admin_index]}")

    def draw_ar_object(self, pid: int, name: str, norm_x: float, norm_y: float, angle: float) -> None:
        ar_objects[pid] = {"name": name, "x": norm_x, "y": norm_y, "angle": angle}
        self.ar_canvas.update()
        log_event(f"AR overlay for marker {pid} ({name}) drawn at ({norm_x:.2f}, {norm_y:.2f}).")

    def remove_ar_object(self, pid: int) -> None:
        if pid in ar_objects:
            ar_objects.pop(pid, None)
            self.ar_canvas.update()
            log_event(f"AR overlay for marker {pid} removed.")

    def update_camera_frame(self, frame) -> None:
        if frame is None:
            return
        self._last_camera_frame = frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.preview_label.setText("")
        self.preview_label.setPixmap(
            pixmap.scaled(
                self.preview_label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self._last_camera_frame is not None:
            self.update_camera_frame(self._last_camera_frame)
        self._fit_text_boxes()


def camera_thread() -> None:
    global latest_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        log_event("Camera failed to open.", level="error")
        return

    last_emit = 0.0
    while not stop_event.is_set():
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)

        with frame_lock:
            latest_frame = frame.copy()

        now = time.time()
        if (now - last_emit) > (1 / 15):
            try:
                signals.camera_frame.emit(frame.copy())
            except RuntimeError:
                break
            last_emit = now

        time.sleep(0.005)

    cap.release()


def handle_socket_client(conn: socket.socket, addr: Tuple[str, int]) -> None:
    log_event(f"Connected to Java source at {addr}.")
    global socket_clients
    with socket_clients_lock:
        socket_clients += 1

    selector_marker = 9
    menu_marker = 10
    payment_marker = 11

    buffer = ""
    with conn:
        while True:
            data = conn.recv(1024)
            if not data:
                break
            try:
                buffer += data.decode()
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    msg = json.loads(line)
                    event = msg.get("event")
                    pid = normalize_marker_id(msg.get("id"))
                    if pid is None:
                        continue
                    pid = marker_alias.get(pid, pid)
                    angle = msg.get("angle", 0)
                    x = msg.get("x", 0.5)
                    y = msg.get("y", 0.5)
                    now = time.time()

                    if event == "add":
                        if pid == menu_marker:
                            if current_ui_state in ("customer", "payment"):
                                signals.set_menu_context.emit("customer")
                            elif current_ui_state == "admin":
                                signals.set_menu_context.emit("admin")
                            elif current_ui_state in ("manage_users", "edit_stock", "purchases", "inventory", "reports"):
                                log_event("Menu marker active in sub-menu.")
                            else:
                                log_event("Ignored menu marker outside active state.")
                        elif pid == selector_marker:
                            if calibration_active:
                                continue
                            if (now - last_select_time.get(pid, 0)) > 1.5:
                                last_select_time[pid] = now
                                if customer_menu_open:
                                    signals.handle_customer_selection.emit()
                                elif manage_users_open:
                                    signals.handle_manage_users_selection.emit()
                                elif edit_stock_open:
                                    signals.handle_edit_stock_selection.emit()
                                elif purchases_open:
                                    signals.handle_purchases_selection.emit()
                                elif inventory_open:
                                    signals.handle_inventory_selection.emit()
                                elif reports_open:
                                    signals.handle_reports_selection.emit()
                                elif admin_menu_open:
                                    signals.handle_admin_selection.emit()
                        elif pid == payment_marker:
                            log_event("Payment marker detected. Completing transaction.")
                            with face_status_lock:
                                payer_name = current_identity["name"]
                                payer_role = current_identity["role"]
                            if payer_role == "customer":
                                success, message = record_purchase(payer_name, cart)
                                if not success:
                                    signals.update_customer_display.emit(message)
                                    signals.show_payment_state.emit(message, False)
                                    continue
                            cart.clear()
                            signals.update_customer_display.emit(
                                "Payment confirmed!\nThank you for shopping with Smart Pharmacy."
                            )
                            signals.show_payment_state.emit(
                                "Payment confirmed!\nThank you for shopping with Smart Pharmacy.",
                                True,
                            )
                        elif pid in product_map:
                            stock_row = get_inventory_stock_by_marker(pid)
                            if stock_row is not None:
                                item_name, stock_qty = stock_row
                                if stock_qty <= 0:
                                    signals.update_customer_display.emit(
                                        f"{item_name} is out of stock."
                                    )
                                    log_event(f"Out of stock: {item_name}.")
                                    continue
                            if pid not in cart:
                                item_info = get_inventory_item_by_marker(pid)
                                item_name = item_info["name"] if item_info else product_map[pid]
                                item_price = item_info["price"] if item_info else 0.0
                                cart[pid] = {
                                    "name": item_name,
                                    "qty": 1,
                                    "price": item_price,
                                }
                                log_event(f"Added {item_name} to cart.")
                            if not admin_menu_open and not manage_users_open:
                                signals.set_ui_state.emit("customer")
                            signals.draw_ar_object.emit(pid, product_map[pid], x, y, angle)

                    elif event == "update":
                        prev_angle = previous_angles.get(pid, angle)
                        diff = angle - prev_angle
                        if diff > math.pi:
                            diff -= 2 * math.pi
                        elif diff < -math.pi:
                            diff += 2 * math.pi
                        previous_angles[pid] = angle
                        update_marker_trajectory(pid, x, y, angle, now)
                        classify_marker_trajectory(pid, now)

                        if pid in product_map:
                            if not admin_menu_open and not manage_users_open:
                                signals.set_ui_state.emit("customer")
                            signals.draw_ar_object.emit(pid, product_map[pid], x, y, angle)

                            if abs(diff) > 0.15 and (now - last_change_time.get(pid, 0)) > 0.4:
                                last_change_time[pid] = now
                                if diff > 0:
                                    signals.rotate_right.emit(pid)
                                else:
                                    signals.rotate_left.emit(pid)
                        elif pid == menu_marker:
                            if abs(diff) > 0.15 and (now - last_change_time.get(pid, 0)) > 0.4:
                                last_change_time[pid] = now
                                if diff > 0:
                                    signals.rotate_right.emit(pid)
                                else:
                                    signals.rotate_left.emit(pid)

                        if pid == selector_marker:
                            if calibration_active:
                                continue
                            if (now - last_select_time.get(pid, 0)) > 1.5:
                                last_select_time[pid] = now
                                if customer_menu_open:
                                    signals.handle_customer_selection.emit()
                                elif manage_users_open:
                                    signals.handle_manage_users_selection.emit()
                                elif edit_stock_open:
                                    signals.handle_edit_stock_selection.emit()
                                elif purchases_open:
                                    signals.handle_purchases_selection.emit()
                                elif inventory_open:
                                    signals.handle_inventory_selection.emit()
                                elif reports_open:
                                    signals.handle_reports_selection.emit()
                                elif admin_menu_open:
                                    signals.handle_admin_selection.emit()

                    elif event == "remove":
                        if pid in product_map:
                            signals.remove_ar_object.emit(pid)
                        marker_paths.pop(pid, None)
                        last_traj_label.pop(pid, None)
                        last_traj_time.pop(pid, None)
            except Exception as e:
                log_event(f"Decode error: {e}", level="error")

    with socket_clients_lock:
        socket_clients = max(0, socket_clients - 1)
    log_event(f"Socket client disconnected: {addr}.")


def socket_thread() -> None:
    HOST, PORT = "127.0.0.1", 5055 #***
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)
    log_event("Socket server listening on 5055.")

    server.settimeout(1.0)
    while not stop_event.is_set():
        try:
            conn, addr = server.accept()
        except socket.timeout:
            continue
        threading.Thread(target=handle_socket_client, args=(conn, addr), daemon=True).start()

    server.close()


def export_attention_results() -> None:
    finalize_customer_session()
    if HEATMAP_DIR.exists():
        log_event(f"Heatmaps saved to {HEATMAP_DIR}")


def main() -> None:
    init_face_db()
    seed_inventory_if_empty()
    face_rec.set_context(sys.modules[__name__])
    face_rec.load_known_users()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    def _shutdown():
        stop_event.set()
        export_attention_results()

    app.aboutToQuit.connect(_shutdown)
    window = MainWindow()
    window.show()

    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=face_rec.face_recognition_thread, daemon=True).start()
    threading.Thread(target=face_rec.emotion_worker, daemon=True).start()
    threading.Thread(target=face_rec.gaze_tracking_thread, daemon=True).start()
    threading.Thread(target=socket_thread, daemon=True).start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
