BG = "#f5f5f7"
CARD = "#ffffff"
CARD_SOFT = "#f2f2f7"
ACCENT = "#0a84ff"
TEXT = "#111111"
MUTED = "#4f4f54"
BORDER = "#e5e5ea"

APP_STYLE = """
QWidget {
    background-color: #f5f5f7;
    color: #111111;
    font-family: "SF Pro Display", "Segoe UI", "Helvetica Neue";
    font-size: 13px;
}
QWidget#app {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 #f7f7fa, stop:1 #eef0f4);
}
QFrame[role="card"] {
    background-color: #ffffff;
    border: 1px solid #ededf0;
    border-radius: 18px;
}
QFrame[role="soft-card"] {
    background-color: #f2f2f7;
    border: 1px solid #ededf0;
    border-radius: 18px;
}
QFrame[role="preview"] {
    background-color: #f2f2f7;
    border: 1px solid #ededf0;
    border-radius: 16px;
}
QFrame[role="overlay"] {
    background-color: transparent;
}
QLabel[role="title"] {
    font-size: 24px;
    font-weight: 600;
}
QLabel[role="subtitle"] {
    font-size: 12px;
    color: #4f4f54;
}
QLabel[role="card-title"] {
    font-size: 14px;
    font-weight: 600;
}
QLabel[role="muted"] {
    color: #4f4f54;
    font-size: 12px;
}
QLabel[role="status"] {
    color: #2c2c2e;
    font-size: 11px;
}
QLabel[role="error"] {
    color: #b42318;
    font-size: 11px;
}
QLabel[role="controls"] {
    color: #4f4f54;
    font-size: 10px;
}
QLabel[role="hero"] {
    font-size: 22px;
    font-weight: 600;
}
QLabel[role="metric"] {
    font-size: 18px;
    font-weight: 600;
    color: #1c1c1e;
}
QLabel[role="badge"] {
    background-color: #edf1f7;
    color: #4f4f54;
    border-radius: 10px;
    padding: 4px 10px;
    font-size: 11px;
}
QPlainTextEdit {
    background-color: transparent;
    border: none;
    padding: 2px 0px;
    color: #1c1c1e;
    font-size: 13px;
}
QLineEdit, QComboBox {
    background-color: #ffffff;
    border: 1px solid #ededf0;
    border-radius: 10px;
    padding: 6px 8px;
    color: #111111;
    font-size: 13px;
}
QPushButton {
    background-color: #0a84ff;
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 8px 14px;
    font-weight: 600;
}
QPushButton:pressed {
    background-color: #0066cc;
}
"""
