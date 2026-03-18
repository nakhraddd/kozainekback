# app/config.py

class AppSettings:
    # Camera Parameters
    CAMERA_DEFAULT_ID = 0  # Default local webcam ID
    CAMERA_RESOLUTION_WIDTH = 640
    CAMERA_RESOLUTION_HEIGHT = 480
    CAMERA_FPS = 30 # Target FPS for local camera capture

    # Detector Parameters
    DETECTOR_MODEL_PATH = "yolov8n-seg.pt" # YOLO model path
    DETECTOR_CONF_THRESHOLD = 0.5 # Confidence threshold for detections
    DETECTOR_FOCAL_LENGTH = 1680.0 # Focal length for distance estimation
    DETECTOR_IMG_SIZE = 640 # Image size for YOLO inference
    # Voice Parameters
    VOICE_RATE = 0 # Default speech rate (0 is normal)
    VOICE_VOLUME = 100 # Default speech volume (0-100)

    # Announcement Intervals (in seconds)
    ANNOUNCEMENT_MIN_INTERVAL = 4 # Minimum interval between repetitions of the same object
    ANNOUNCEMENT_AUTO_INTERVAL = 6 # Automatic announcement interval (for blind user flow)

    # Server/Client Communication
    SERVER_HOST = "localhost"
    SERVER_PORT = 8000
    SERVER_HTTP_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
    DETECTION_WEBSOCKET_URL = f"ws://{SERVER_HOST}:{SERVER_PORT}/ws"
    # NGROK_DOMAIN = "anemone-intimate-utterly.ngrok-free.app" # Your ngrok domain if used

    # Visualization Settings
    VISUALIZATION_ALPHA = 0.3 # Transparency for masks
    VISUALIZATION_FONT_SIZE = 20 # Font size for labels
    VISUALIZATION_FONT_PATH = "arial.ttf" # Font for Cyrillic support

    # Paths
    SCREENSHOT_FOLDER = "screenshots/"

# Create an instance of settings to be imported
settings = AppSettings()
