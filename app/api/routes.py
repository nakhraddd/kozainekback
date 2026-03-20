import cv2
import numpy as np
import logging
import json
import uuid
import asyncio
import os
import sys
import shutil
import requests
import websockets
from urllib.parse import urlparse, parse_qs
from typing import Dict

from app.services.detector import ObjectDetector, YoloDetector
from app.domain.logic import SpatialAnalyzer
from app.domain.priorities import HIGH_PRIORITY_OBJECTS, MEDIUM_PRIORITY_OBJECTS
from app.domain.message_formatter import format_message, RUSSIAN_NAMES
from app.services.voice_output import VoiceAssistant
from app.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Global State for Camera Management ---
ACTIVE_CLIENTS: Dict[str, websockets.ServerConnection] = {}
LATEST_FRAMES: Dict[str, bytes] = {}


def get_cameras():
    """Direct memory access for CameraService, replacing the old HTTP endpoint."""
    cameras = [{"id": "local_0", "name": "Local Server Webcam", "type": "local"}]
    for client_id in ACTIVE_CLIENTS:
        cameras.append({
            "id": client_id,
            "name": f"Remote Client {client_id[:8]}",
            "type": "remote_websocket",
            "stream_ws_url": f"ws://localhost:8000/ws/view/{client_id}"
        })
    return cameras


def get_priority_level(english_name: str) -> int:
    if english_name in HIGH_PRIORITY_OBJECTS:
        return 3
    elif english_name in MEDIUM_PRIORITY_OBJECTS:
        return 2
    else:
        return 1


class ConnectionManager:
    def __init__(self, detector: ObjectDetector):
        self.detector = detector
        self.voice_assistant = VoiceAssistant()
        self.previous_tracked_objects = {}
        self.previous_static_objects = {}
        self.last_processed_frame = {}
        self.last_detection_result = {}
        self.current_language = "RUSSIAN"
        self.client_states = {}

    async def handle_ws(self, websocket):
        client_id = str(uuid.uuid4())
        ACTIVE_CLIENTS[client_id] = websocket

        # Parse query parameters from the raw WebSocket URI
        parsed_path = urlparse(websocket.request.path)
        query_params = parse_qs(parsed_path.query)
        query_lang = query_params.get("lang", ["RUSSIAN"])[0]
        initial_lang = query_lang if query_lang in ["ENGLISH", "RUSSIAN", "KAZAKH"] else "RUSSIAN"

        self.client_states[client_id] = {"language": initial_lang}
        self.previous_tracked_objects[client_id] = set()
        self.previous_static_objects[client_id] = set()
        self.last_processed_frame[client_id] = None
        self.last_detection_result[client_id] = {"text": "", "boxes": []}

        logger.info(f"Client connected with ID: {client_id}, Lang: {initial_lang}")
        last_log_text = ""

        new_frame_event = asyncio.Event()
        running = True

        async def receive_loop():
            nonlocal running
            try:
                while running:
                    # Pure websockets recv() returns bytes for binary frames, str for text
                    message = await asyncio.wait_for(websocket.recv(), timeout=10.0)

                    if isinstance(message, bytes):
                        LATEST_FRAMES[client_id] = message
                        new_frame_event.set()
                    elif isinstance(message, str):
                        try:
                            data = json.loads(message)
                            if data.get("action") == "set_language":
                                new_lang = data.get("language")
                                if new_lang in ["ENGLISH", "RUSSIAN", "KAZAKH"]:
                                    self.client_states[client_id]["language"] = new_lang
                                    logger.info(f"Client {client_id} changed language to {new_lang}")
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from client {client_id}")

            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
                logger.warning(f"Client {client_id} disconnected or timed out.")
            finally:
                running = False
                new_frame_event.set()

        receiver_task = asyncio.create_task(receive_loop())

        try:
            while running:
                await new_frame_event.wait()
                new_frame_event.clear()

                if not running: break

                data = LATEST_FRAMES.get(client_id)
                if data is None: continue

                np_arr = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None: continue

                last_frame = self.last_processed_frame.get(client_id)
                if last_frame is not None:
                    diff = cv2.absdiff(frame, last_frame)
                    change_percentage = np.count_nonzero(diff) / frame.size
                    if change_percentage < 0.1:
                        if running: await websocket.send(json.dumps(self.last_detection_result[client_id]))
                        continue
                    elif change_percentage > 0.8:
                        self.previous_tracked_objects[client_id].clear()
                        self.previous_static_objects[client_id].clear()

                def process_frame_logic():
                    h, w, _ = frame.shape
                    analyzer = SpatialAnalyzer(frame_width=w, frame_height=h)
                    raw_detections = self.detector.detect(frame)

                    if not raw_detections:
                        return {"text": "", "boxes": []}

                    newly_spoken_objects = []
                    current_tracked = set()
                    current_static = set()
                    all_detected_objects = []

                    for raw_det in raw_detections:
                        processed_obj = analyzer.analyze(raw_det)
                        all_detected_objects.append({"processed": processed_obj, "raw": raw_det})

                        if processed_obj.track_id is not None:
                            current_tracked.add(processed_obj.track_id)
                            if processed_obj.track_id not in self.previous_tracked_objects.get(client_id, set()):
                                newly_spoken_objects.append(processed_obj)
                        else:
                            current_static.add(processed_obj.name)
                            if processed_obj.name not in self.previous_static_objects.get(client_id, set()):
                                newly_spoken_objects.append(processed_obj)

                    self.previous_tracked_objects[client_id] = current_tracked
                    self.previous_static_objects[client_id] = current_static
                    self.last_processed_frame[client_id] = frame.copy()

                    client_lang = self.client_states.get(client_id, {}).get("language", "RUSSIAN")
                    text_result = format_message(newly_spoken_objects, language=client_lang)

                    response_boxes = []
                    for item in all_detected_objects:
                        proc_obj = item["processed"]
                        raw_obj = item["raw"]
                        english_name = next((en for en, ru in RUSSIAN_NAMES.items() if ru == proc_obj.name),
                                            proc_obj.name)
                        priority = get_priority_level(english_name)

                        response_boxes.append({
                            "name": proc_obj.name,
                            "xmin": round(proc_obj.normalized_box[0], 4), "ymin": round(proc_obj.normalized_box[1], 4),
                            "xmax": round(proc_obj.normalized_box[2], 4), "ymax": round(proc_obj.normalized_box[3], 4),
                            "distance_cm": round(proc_obj.distance_cm, 2) if proc_obj.distance_cm is not None else None,
                            "confidence": round(raw_obj.confidence, 4),
                            "priority": priority,
                            "mask_points": [[round(p, 4) for p in point] for point in
                                            proc_obj.normalized_mask_points] if proc_obj.normalized_mask_points else None,
                            "track_id": proc_obj.track_id
                        })

                    return {"text": text_result, "boxes": response_boxes}

                # Replaced Starlette's threadpool with standard asyncio
                result = await asyncio.to_thread(process_frame_logic)

                if running:
                    if result:
                        self.last_detection_result[client_id] = result
                        if result["text"] and result["text"] != last_log_text:
                            logger.info(f"Detected for {client_id[:8]}: {result['text']}")
                            last_log_text = result["text"]

                        await websocket.send(json.dumps(result))
                    else:
                        await websocket.send(json.dumps(self.last_detection_result[client_id]))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket disconnected for client {client_id}.")
        except Exception as e:
            if running:
                logger.error(f"Error in processor loop for {client_id}: {e}", exc_info=True)
        finally:
            running = False
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass

            if client_id in ACTIVE_CLIENTS: del ACTIVE_CLIENTS[client_id]
            if client_id in LATEST_FRAMES: del LATEST_FRAMES[client_id]
            if client_id in self.previous_tracked_objects: del self.previous_tracked_objects[client_id]
            if client_id in self.previous_static_objects: del self.previous_static_objects[client_id]
            if client_id in self.last_processed_frame: del self.last_processed_frame[client_id]
            if client_id in self.last_detection_result: del self.last_detection_result[client_id]
            if client_id in self.client_states: del self.client_states[client_id]
            logger.info(f"Client {client_id} disconnected and cleaned up.")


# --- Model Loading Logic ---
if getattr(sys, 'frozen', False):
    bundle_dir = sys._MEIPASS
    model_file_name = settings.DETECTOR_MODEL_PATH
    model_path_to_use = os.path.join(bundle_dir, model_file_name)
    if not os.path.exists(model_path_to_use):
        exe_dir = os.path.dirname(sys.executable)
        model_path_to_use = os.path.join(exe_dir, model_file_name)
else:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    bundle_dir = project_root
    model_file_name = settings.DETECTOR_MODEL_PATH
    model_path_to_use = os.path.join(bundle_dir, model_file_name)


def download_file(url, filename):
    try:
        logger.info(f"Downloading model from {url} to {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info(f"Downloaded {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {filename}: {e}")
        return False


if not os.path.exists(model_path_to_use):
    logger.info(f"Model not found at {model_path_to_use}. Attempting download...")
    model_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-seg.pt"
    download_file(model_url, model_path_to_use)

detector = YoloDetector(model_path=model_path_to_use)
manager = ConnectionManager(detector=detector)