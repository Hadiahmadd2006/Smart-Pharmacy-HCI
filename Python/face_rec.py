import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from deepface import DeepFace

from db import db_session
_context = None


def set_context(context) -> None:
    global _context
    _context = context


def _ctx():
    if _context is None:
        raise RuntimeError("Vision context is not initialized.")
    return _context


def build_embedding_from_image(image_bgr: np.ndarray) -> Optional[np.ndarray]:
    ctx = _ctx()
    try:
        result = DeepFace.represent(
            image_bgr,
            model_name=ctx.FACE_MODEL,
            enforce_detection=False,
        )
        if isinstance(result, list):
            result = result[0]
        embedding = np.array(result["embedding"], dtype=np.float32)
        return embedding
    except Exception as exc:
        ctx.log_event(f"DeepFace represent failed: {exc}", level="error")
        return None


def cosine_distance(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(vec_a, vec_b) / denom)


def find_best_match(embedding: np.ndarray) -> tuple[Optional[dict], float]:
    ctx = _ctx()
    with ctx.known_users_lock:
        users = list(ctx.known_users)
    if not users:
        return None, 1.0

    best_user = None
    best_distance = 1.0
    for user in users:
        distance = cosine_distance(embedding, user["embedding"])
        if distance < best_distance:
            best_distance = distance
            best_user = user

    return best_user, best_distance


def load_known_users() -> None:
    ctx = _ctx()
    users = []
    roles_by_name: dict[str, set[str]] = {}
    with db_session() as conn:
        cursor = conn.execute("SELECT name, role, embedding, dim FROM faces")
        for name, role, embedding_blob, dim in cursor.fetchall():
            embedding = np.frombuffer(embedding_blob, dtype=np.float32, count=dim)
            users.append({"name": name, "role": role, "embedding": embedding})
            roles_by_name.setdefault(name, set()).add(role)
    with ctx.known_users_lock:
        ctx.known_users.clear()
        ctx.known_users.extend(users)
    conflicted = {name for name, roles in roles_by_name.items() if len(roles) > 1}
    ctx.conflicted_names.clear()
    ctx.conflicted_names.update(conflicted)
    ctx.log_event(f"Loaded {len(users)} known face embeddings.")


def emotion_worker() -> None:
    ctx = _ctx()
    last_time = 0.0
    ctx.log_event("Emotion worker started.")

    while not ctx.stop_event.is_set():
        with ctx.frame_lock:
            frame = ctx.latest_frame.copy() if ctx.latest_frame is not None else None

        if frame is None:
            time.sleep(0.1)
            continue

        now = time.time()
        if (now - last_time) > 1.0:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = DeepFace.analyze(
                    rgb,
                    actions=["emotion"],
                    enforce_detection=False,
                    silent=True,
                )
                if isinstance(result, list):
                    result = result[0]
                with ctx.face_status_lock:
                    ctx.current_emotion = result.get("dominant_emotion", "Unknown")
                ctx.update_face_status_text()
            except Exception as exc:
                ctx.log_event(f"Emotion analyze failed: {exc}", level="error")
            last_time = now

        time.sleep(0.05)


def face_recognition_thread() -> None:
    ctx = _ctx()
    last_check = 0.0
    last_name = None
    missing_since = None
    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.3,
    )
    ctx.log_event("Face recognition worker started.")

    def session_active() -> bool:
        with ctx.face_status_lock:
            role = ctx.current_identity["role"]
        return role in ("admin", "customer")

    def has_visible_face(result: Optional[object]) -> bool:
        if not result or not getattr(result, "detections", None):
            return False
        for det in result.detections:
            score = max(det.score) if getattr(det, "score", None) else 0.0
            bbox = det.location_data.relative_bounding_box
            area = float(bbox.width * bbox.height)
            if score >= 0.6 and area >= 0.04:
                return True
        return False

    def should_route_to(role: str) -> bool:
        state = ctx.current_ui_state
        admin_states = {"admin", "reports", "manage_users", "edit_stock", "purchases", "inventory"}
        customer_states = {"customer", "payment"}
        if role == "admin":
            return state not in admin_states
        if role == "customer":
            return state not in customer_states
        return False

    while not ctx.stop_event.is_set():
        with ctx.frame_lock:
            frame = ctx.latest_frame.copy() if ctx.latest_frame is not None else None

        if frame is None:
            time.sleep(0.1)
            continue

        now = time.time()
        if (now - last_check) < 1.2:
            time.sleep(0.05)
            continue
        last_check = now

        with ctx.known_users_lock:
            has_users = bool(ctx.known_users)
        if not has_users:
            if not session_active():
                with ctx.face_status_lock:
                    ctx.current_identity["name"] = "No enrolled faces"
                    ctx.current_identity["role"] = "unknown"
                    ctx.current_identity["confidence"] = 0.0
                ctx.update_face_status_text()
                ctx.signals.set_ui_state.emit("idle")
            missing_since = None
            time.sleep(1.0)
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = face_detector.process(rgb)
        has_face = has_visible_face(detection)

        if not has_face:
            if session_active():
                if missing_since is None:
                    missing_since = now
                    ctx.log_event("No face detected. Starting idle timeout.")
                if (now - missing_since) >= ctx.FACE_MISSING_TIMEOUT:
                    with ctx.face_status_lock:
                        ctx.current_identity["name"] = "Recognizing user"
                        ctx.current_identity["role"] = "unknown"
                        ctx.current_identity["confidence"] = 0.0
                    ctx.update_face_status_text()
                    ctx.signals.set_ui_state.emit("idle")
                    last_name = None
                    missing_since = None
            else:
                missing_since = None
            time.sleep(0.2)
            continue

        embedding = build_embedding_from_image(frame)
        if embedding is None:
            if session_active():
                if missing_since is None:
                    missing_since = now
                    ctx.log_event("Face embedding unavailable. Starting idle timeout.")
                if (now - missing_since) >= ctx.FACE_MISSING_TIMEOUT:
                    with ctx.face_status_lock:
                        ctx.current_identity["name"] = "Recognizing user"
                        ctx.current_identity["role"] = "unknown"
                        ctx.current_identity["confidence"] = 0.0
                    ctx.update_face_status_text()
                    ctx.signals.set_ui_state.emit("idle")
                    last_name = None
                    missing_since = None
            else:
                missing_since = None
            time.sleep(0.2)
            continue

        match, distance = find_best_match(embedding)
        if match and distance <= ctx.FACE_MATCH_THRESHOLD:
            if match["name"] in ctx.conflicted_names:
                ctx.signals.update_auth_error.emit(
                    "Error: user is enrolled as both customer and admin.\nFix roles to continue."
                )
                if not session_active():
                    with ctx.face_status_lock:
                        ctx.current_identity["name"] = match["name"]
                        ctx.current_identity["role"] = "conflict"
                        ctx.current_identity["confidence"] = 0.0
                    ctx.update_face_status_text()
                    ctx.signals.set_ui_state.emit("idle")
                    last_name = None
                    missing_since = None
                time.sleep(0.4)
                continue

            ctx.signals.update_auth_error.emit("")
            confidence = max(0.0, 1.0 - (distance / ctx.FACE_MATCH_THRESHOLD))
            if match["role"] == "customer":
                if ctx.active_customer_name and ctx.active_customer_name != match["name"]:
                    ctx.finalize_customer_session()
                if ctx.active_customer_name != match["name"]:
                    ctx.start_customer_session(match["name"])
            elif match["role"] == "admin":
                if ctx.active_customer_name:
                    ctx.finalize_customer_session()
            with ctx.face_status_lock:
                ctx.current_identity["name"] = match["name"]
                ctx.current_identity["role"] = match["role"]
                ctx.current_identity["confidence"] = confidence
            ctx.update_face_status_text()
            if match["role"] == "admin":
                if should_route_to("admin"):
                    ctx.signals.set_ui_state.emit("admin")
            else:
                if should_route_to("customer"):
                    ctx.signals.set_ui_state.emit("customer")
            if match["name"] != last_name:
                ctx.log_event(f"Recognized {match['name']} ({match['role']}).")
            last_name = match["name"]
            missing_since = None
        else:
            ctx.signals.update_auth_error.emit("")
            if session_active():
                if missing_since is None:
                    missing_since = now
                    ctx.log_event("Face mismatch. Starting idle timeout.")
                if (now - missing_since) >= ctx.FACE_MISSING_TIMEOUT:
                    with ctx.face_status_lock:
                        ctx.current_identity["name"] = "Recognizing user"
                        ctx.current_identity["role"] = "unknown"
                        ctx.current_identity["confidence"] = 0.0
                    ctx.update_face_status_text()
                    ctx.signals.set_ui_state.emit("idle")
                    last_name = None
                    missing_since = None
            else:
                with ctx.face_status_lock:
                    ctx.current_identity["name"] = "Unknown"
                    ctx.current_identity["role"] = "unknown"
                    ctx.current_identity["confidence"] = 0.0
                ctx.update_face_status_text()
                ctx.signals.set_ui_state.emit("idle")
                last_name = None


def gaze_tracking_thread() -> None:
    ctx = _ctx()
    ctx.log_event("Gaze tracking worker started.")

    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.3,
    )
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    heatmap_radius = 22
    kernel_size = heatmap_radius * 2 + 1
    kernel_sigma = max(2.0, heatmap_radius / 2.3)
    kernel = cv2.getGaussianKernel(kernel_size, kernel_sigma)
    kernel = kernel @ kernel.T
    kernel = kernel / kernel.max()
    kernel_radius_y = kernel.shape[0] // 2
    kernel_radius_x = kernel.shape[1] // 2
    smooth_alpha_x = 0.18
    smooth_alpha_y = 0.20
    max_jump_x = 0.12
    max_jump_y = 0.14
    raw_smooth_alpha = 0.12
    use_face_fallback = False
    auto_window = 120
    auto_min_samples = 6
    auto_low_x = 15
    auto_high_x = 85
    auto_low_y = 10
    auto_high_y = 90
    auto_min_span_x = 0.10
    auto_min_span_y = 0.10
    auto_default_low_x = 0.2
    auto_default_high_x = 0.8
    auto_default_low_y = 0.15
    auto_default_high_y = 0.85
    auto_expand_x = 1.45
    auto_freeze_seconds = 4.0
    gaze_x_gain = 2.2
    gaze_y_gain = 1.0

    while not ctx.stop_event.is_set():
        with ctx.frame_lock:
            frame = ctx.latest_frame.copy() if ctx.latest_frame is not None else None

        if frame is None:
            time.sleep(0.02)
            continue

        with ctx.session_lock:
            active_name = ctx.active_customer_name
            active = ctx.session_active
        if ctx.current_ui_state != "customer":
            time.sleep(0.02)
            continue

        h, w, _ = frame.shape
        target_h, target_w = h, w
        if ctx.heatmap_target_size:
            target_h, target_w = ctx.heatmap_target_size
        with ctx.session_lock:
            if ctx.session_heatmap_acc is None or ctx.session_heatmap_acc.shape != (target_h, target_w):
                ctx.session_heatmap_acc = np.zeros((target_h, target_w), dtype=np.float32)
            ctx.heatmap_acc = ctx.session_heatmap_acc

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_result = face_mesh.process(rgb)

        gx = None
        gy = None
        gaze_ratio = None
        eye_open = None
        yaw = 0.0
        pitch = 0.0
        stable = False

        if mesh_result.multi_face_landmarks:
            landmarks = mesh_result.multi_face_landmarks[0].landmark

            def _avg_point(indices: list[int]) -> tuple[float, float]:
                xs = [landmarks[i].x for i in indices]
                ys = [landmarks[i].y for i in indices]
                return float(sum(xs) / len(xs)), float(sum(ys) / len(ys))

            left_iris = [468, 469, 470, 471]
            right_iris = [473, 474, 475, 476]

            lx, ly = _avg_point(left_iris)
            rx, ry = _avg_point(right_iris)

            left_outer = landmarks[33].x
            left_inner = landmarks[133].x
            right_inner = landmarks[362].x
            right_outer = landmarks[263].x

            left_top = landmarks[159].y
            left_bottom = landmarks[145].y
            right_top = landmarks[386].y
            right_bottom = landmarks[374].y

            left_span_x = max(1e-6, float(left_inner - left_outer))
            right_span_x = max(1e-6, float(right_outer - right_inner))
            left_span_y = max(1e-6, float(left_bottom - left_top))
            right_span_y = max(1e-6, float(right_bottom - right_top))

            left_x_ratio = float((lx - left_outer) / left_span_x)
            right_x_ratio = float((rx - right_inner) / right_span_x)
            left_y_ratio = float((ly - left_top) / left_span_y)
            right_y_ratio = float((ry - right_top) / right_span_y)

            nose = landmarks[1]
            nose_x = float(nose.x)
            nose_y = float(nose.y)
            eye_span = max(1e-6, float(right_outer - left_outer))
            mid_eye_x = (left_outer + right_outer) / 2
            mid_eye_y = (left_top + right_top) / 2
            yaw = float((nose_x - mid_eye_x) / eye_span)
            pitch = float((nose_y - mid_eye_y) / eye_span)
            yaw = max(-0.6, min(0.6, yaw))
            pitch = max(-0.6, min(0.6, pitch))

            eye_open = (left_span_y + right_span_y) / 2
            if eye_open < 0.008:
                gaze_x = None
                gaze_y = None
            else:
                if getattr(ctx, "GAZE_FLIP_X", False):
                    left_x_ratio = 1.0 - left_x_ratio
                    right_x_ratio = 1.0 - right_x_ratio
                    yaw = -yaw
                if getattr(ctx, "GAZE_FLIP_Y", False):
                    left_y_ratio = 1.0 - left_y_ratio
                    right_y_ratio = 1.0 - right_y_ratio
                    pitch = -pitch

                gaze_x_raw = (left_x_ratio + right_x_ratio) / 2
                gaze_y_raw = (left_y_ratio + right_y_ratio) / 2
                if not hasattr(ctx, "gaze_x_center"):
                    ctx.gaze_x_center = gaze_x_raw
                else:
                    ctx.gaze_x_center = ctx.gaze_x_center * 0.98 + gaze_x_raw * 0.02
                gaze_x = ctx.gaze_x_center + (gaze_x_raw - ctx.gaze_x_center) * gaze_x_gain
                gaze_y = 0.5 + (gaze_y_raw - 0.5) * gaze_y_gain
                gaze_x = max(0.0, min(1.0, gaze_x))
                gaze_y = max(0.0, min(1.0, gaze_y))

                gx = gaze_x
                gy = gaze_y
                ctx.gaze_raw_x = gaze_x_raw
                ctx.gaze_raw_y = gaze_y_raw
                gaze_ratio = gaze_y
                ctx.gaze_features = (
                    left_x_ratio,
                    right_x_ratio,
                    left_y_ratio,
                    right_y_ratio,
                    yaw,
                    pitch,
                )
                ctx.gaze_features_timestamp = time.time()
                ctx.gaze_stable = True
                ctx.gaze_stable_timestamp = time.time()

        if gx is None or gy is None:
            if use_face_fallback:
                result = face_detector.process(rgb)
                if result.detections:
                    det = result.detections[0]
                    bbox = det.location_data.relative_bounding_box
                    gx = max(0.0, min(1.0, bbox.xmin + bbox.width / 2))
                    gy = max(0.0, min(1.0, bbox.ymin + bbox.height / 2))
            if gx is None or gy is None:
                ctx.gaze_stable = False
                ctx.gaze_stable_timestamp = time.time()

        if gx is None or gy is None:
            time.sleep(0.02)
            continue

        stable = eye_open is not None and eye_open >= 0.006
        ctx.gaze_stable = stable
        ctx.gaze_stable_timestamp = time.time()

        raw_gx = getattr(ctx, "gaze_raw_x", gx)
        raw_gy = getattr(ctx, "gaze_raw_y", gy)
        if not hasattr(ctx, "gaze_raw_smooth"):
            ctx.gaze_raw_smooth = (raw_gx, raw_gy)
        else:
            last_rx, last_ry = ctx.gaze_raw_smooth
            smooth_rx = last_rx + (raw_gx - last_rx) * raw_smooth_alpha
            smooth_ry = last_ry + (raw_gy - last_ry) * raw_smooth_alpha
            ctx.gaze_raw_smooth = (smooth_rx, smooth_ry)
        raw_gx, raw_gy = ctx.gaze_raw_smooth
        ctx.gaze_point_norm = (raw_gx, raw_gy)
        ctx.gaze_point_timestamp = time.time()

        mapped = None
        if getattr(ctx, "GAZE_AUTO_MODE", False):
            if not hasattr(ctx, "gaze_auto_x"):
                ctx.gaze_auto_x = deque(maxlen=auto_window)
                ctx.gaze_auto_y = deque(maxlen=auto_window)
                ctx.gaze_auto_ready = False
                ctx.gaze_auto_frozen = False
                ctx.gaze_auto_start = time.time()
            if stable:
                if not ctx.gaze_auto_frozen:
                    ctx.gaze_auto_x.append(raw_gx)
                    ctx.gaze_auto_y.append(raw_gy)
                if ctx.gaze_auto_frozen and hasattr(ctx, "gaze_auto_bounds"):
                    low_x, high_x, low_y, high_y = ctx.gaze_auto_bounds
                else:
                    if len(ctx.gaze_auto_x) >= auto_min_samples:
                        low_x, high_x = np.percentile(ctx.gaze_auto_x, [auto_low_x, auto_high_x])
                    else:
                        low_x, high_x = auto_default_low_x, auto_default_high_x
                    if len(ctx.gaze_auto_y) >= auto_min_samples:
                        low_y, high_y = np.percentile(ctx.gaze_auto_y, [auto_low_y, auto_high_y])
                    else:
                        low_y, high_y = auto_default_low_y, auto_default_high_y
                span_x = float(high_x - low_x)
                span_y = float(high_y - low_y)
                if span_x < auto_min_span_x:
                    mean_x = float(np.mean(ctx.gaze_auto_x)) if ctx.gaze_auto_x else 0.5
                    low_x = mean_x - auto_min_span_x / 2
                    high_x = mean_x + auto_min_span_x / 2
                    span_x = high_x - low_x
                if span_y < auto_min_span_y:
                    mean_y = float(np.mean(ctx.gaze_auto_y)) if ctx.gaze_auto_y else 0.5
                    low_y = mean_y - auto_min_span_y / 2
                    high_y = mean_y + auto_min_span_y / 2
                    span_y = high_y - low_y
                mx = (raw_gx - low_x) / max(1e-3, span_x)
                mx = 0.5 + (mx - 0.5) * auto_expand_x
                my = (raw_gy - low_y) / max(1e-3, span_y)
                mapped = (max(0.0, min(1.0, mx)), max(0.0, min(1.0, my)))
                ctx.gaze_auto_ready = True
                if (
                    not ctx.gaze_auto_frozen
                    and len(ctx.gaze_auto_x) >= auto_min_samples
                    and (time.time() - ctx.gaze_auto_start) >= auto_freeze_seconds
                ):
                    ctx.gaze_auto_bounds = (low_x, high_x, low_y, high_y)
                    ctx.gaze_auto_frozen = True
        if mapped is None:
            mapped = ctx.map_gaze_point(raw_gx, raw_gy)

        gx, gy = mapped
        if not hasattr(ctx, "gaze_mapped_smooth"):
            ctx.gaze_mapped_smooth = (gx, gy)
        else:
            last_x, last_y = ctx.gaze_mapped_smooth
            dx = gx - last_x
            dy = gy - last_y
            if abs(dx) > max_jump_x:
                gx = last_x + max(-max_jump_x, min(max_jump_x, dx))
            if abs(dy) > max_jump_y:
                gy = last_y + max(-max_jump_y, min(max_jump_y, dy))
            smooth_x = last_x + (gx - last_x) * smooth_alpha_x
            smooth_y = last_y + (gy - last_y) * smooth_alpha_y
            ctx.gaze_mapped_smooth = (smooth_x, smooth_y)
            gx, gy = ctx.gaze_mapped_smooth
        ctx.gaze_point_mapped = (gx, gy)
        ctx.gaze_point_mapped_timestamp = time.time()

        # stable is computed before auto mapping for this frame
        cx = int(gx * target_w)
        cy = int(gy * target_h)

        if gx < 0.3:
            zone = "MENU"
        elif gx > 0.7:
            zone = "CART"
        else:
            zone = "AR"

        cx = max(0, min(cx, target_w - 1))
        cy = max(0, min(cy, target_h - 1))
        if (
            stable
            and active
            and active_name
            and not ctx.calibration_active
            and getattr(ctx, "gaze_calibrated", False)
            and getattr(ctx, "gaze_auto_ready", True)
        ):
            with ctx.session_lock:
                if ctx.session_heatmap_acc is None or ctx.session_heatmap_acc.shape != (target_h, target_w):
                    ctx.session_heatmap_acc = np.zeros((target_h, target_w), dtype=np.float32)
                y1 = max(0, cy - kernel_radius_y)
                x1 = max(0, cx - kernel_radius_x)
                y2 = min(target_h, cy + kernel_radius_y + 1)
                x2 = min(target_w, cx + kernel_radius_x + 1)
                ky1 = y1 - (cy - kernel_radius_y)
                kx1 = x1 - (cx - kernel_radius_x)
                ky2 = ky1 + (y2 - y1)
                kx2 = kx1 + (x2 - x1)
                ctx.session_heatmap_acc[y1:y2, x1:x2] += kernel[ky1:ky2, kx1:kx2] * 20

        if (
            stable
            and gaze_ratio is not None
            and active
            and active_name
            and not ctx.calibration_active
            and getattr(ctx, "gaze_calibrated", False)
            and getattr(ctx, "gaze_auto_ready", True)
        ):
            ctx.gaze_scroll_value = gy
            ctx.gaze_scroll_timestamp = time.time()

        with ctx.face_status_lock:
            emotion = ctx.current_emotion
        if (
            stable
            and active
            and active_name
            and not ctx.calibration_active
            and getattr(ctx, "gaze_calibrated", False)
            and getattr(ctx, "gaze_auto_ready", True)
        ):
            ctx.gaze_log.append((cx, cy, zone, emotion, time.time()))

        time.sleep(0.02)
