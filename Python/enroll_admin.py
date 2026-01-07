import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from deepface import DeepFace

from db import init_database, export_database_sql, db_session
from GUI import APP_STYLE

ROLE = "admin"
FACE_MODEL = "Facenet512"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "database"
FACE_IMAGE_DIR = BASE_DIR / "faces"
DATABASE_SQL_PATH = DATA_DIR / "database.sql"

latest_frame = None
frame_lock = threading.Lock()


def build_embedding_from_image(image_bgr: np.ndarray) -> np.ndarray | None:
    try:
        result = DeepFace.represent(
            image_bgr,
            model_name=FACE_MODEL,
            enforce_detection=False,
        )
        if isinstance(result, list):
            result = result[0]
        return np.array(result["embedding"], dtype=np.float32)
    except Exception:
        return None


def upsert_face_embedding(name: str, role: str, embedding: np.ndarray) -> None:
    blob = embedding.astype(np.float32).tobytes()
    dim = int(embedding.size)
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO faces (name, role, embedding, dim, model, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(name, role) DO UPDATE SET
                embedding = excluded.embedding,
                dim = excluded.dim,
                model = excluded.model,
                created_at = excluded.created_at
            """,
            (name, role, blob, dim, FACE_MODEL, time.strftime("%Y-%m-%d %H:%M:%S")),
        )
    export_database_sql(DATABASE_SQL_PATH)


class UiSignals(QtCore.QObject):
    camera_frame = QtCore.Signal(object)


signals = UiSignals()


class EnrollWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Enroll Admin Face")
        self.setMinimumSize(720, 520)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(18)

        card = QtWidgets.QFrame()
        card.setProperty("role", "card")
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(20, 18, 20, 18)
        card_layout.setSpacing(12)

        title = QtWidgets.QLabel("Enroll Admin")
        title.setProperty("role", "title")
        note = QtWidgets.QLabel("Enter a name and capture the current camera frame.")
        note.setProperty("role", "muted")
        note.setWordWrap(True)

        self.name_input = QtWidgets.QLineEdit()
        self.name_input.setPlaceholderText("Admin name")

        self.status = QtWidgets.QLabel("Waiting for input.")
        self.status.setProperty("role", "muted")
        self.status.setWordWrap(True)

        self.preview = QtWidgets.QLabel("Camera preview")
        self.preview.setAlignment(QtCore.Qt.AlignCenter)
        self.preview.setMinimumHeight(220)

        capture_btn = QtWidgets.QPushButton("Capture & Save")
        capture_btn.clicked.connect(self.capture_face)

        card_layout.addWidget(title)
        card_layout.addWidget(note)
        card_layout.addWidget(self.name_input)
        card_layout.addWidget(self.preview)
        card_layout.addWidget(capture_btn)
        card_layout.addWidget(self.status)
        layout.addWidget(card)

        signals.camera_frame.connect(self.update_preview)

    def update_preview(self, frame) -> None:
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def capture_face(self) -> None:
        name = self.name_input.text().strip()
        if not name:
            self.status.setText("Enter a name before capturing.")
            return

        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            self.status.setText("Camera frame not available yet.")
            return

        safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
        timestamp = int(time.time())
        FACE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        image_path = FACE_IMAGE_DIR / f"{ROLE}__{safe_name}_{timestamp}.jpg"

        if not cv2.imwrite(str(image_path), frame):
            self.status.setText("Failed to save face image.")
            return

        embedding = build_embedding_from_image(frame)
        if embedding is None:
            self.status.setText("Face embedding failed. Try again.")
            return

        upsert_face_embedding(name, ROLE, embedding)
        self.status.setText(f"Saved {name} as {ROLE}.")


def camera_thread() -> None:
    global latest_frame
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        return

    last_emit = 0.0
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.01)
            continue

        with frame_lock:
            latest_frame = frame.copy()

        now = time.time()
        if (now - last_emit) > (1 / 15):
            signals.camera_frame.emit(frame.copy())
            last_emit = now

        time.sleep(0.005)


def main() -> None:
    init_database(DATABASE_SQL_PATH)
    export_database_sql(DATABASE_SQL_PATH)
    app = QtWidgets.QApplication([])
    app.setStyleSheet(APP_STYLE)
    window = EnrollWindow()
    window.show()

    threading.Thread(target=camera_thread, daemon=True).start()

    app.exec()


if __name__ == "__main__":
    main()
