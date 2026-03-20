import asyncio
import cv2
import websockets
import json
import numpy as np
import colorsys
from PIL import Image, ImageDraw, ImageFont
import httpx
import logging
import tkinter as tk
import os
from tkinter import filedialog
from concurrent.futures import ThreadPoolExecutor

from app.domain.priorities import HIGH_PRIORITY_OBJECTS, MEDIUM_PRIORITY_OBJECTS, LOW_PRIORITY_OBJECTS
from app.domain.message_formatter import RUSSIAN_NAMES
from app.config import settings

# Configure logging for this service
logger = logging.getLogger(__name__)

# UI Translations
UI_TRANSLATIONS = {
    "RUSSIAN": {
        "masks": "Маски", "boxes": "Рамки", "lang": "Язык", "quit": "Выход",
        "on": "ВКЛ", "off": "ВЫКЛ", "select_src": "Выберите источник камеры:",
        "settings": "Настройки KOZAINEK:",
        "analyse_file": "Анализ фото/видео"
    },
    "ENGLISH": {
        "masks": "Masks", "boxes": "Boxes", "lang": "Lang", "quit": "Quit",
        "on": "ON", "off": "OFF", "select_src": "Select Camera Source:",
        "settings": "KOZAINEK Settings:",
        "analyse_file": "Analyse Photo/Video"
    },
    "KAZAKH": {
        "masks": "Маскалар", "boxes": "Рамкалар", "lang": "Тіл", "quit": "Шығу",
        "on": "ҚОСУ", "off": "ӨШІРУ", "select_src": "Камера көзін таңдаңыз:",
        "settings": "KOZAINEK Баптаулары:",
        "analyse_file": "Фото/Бейне талдау"
    }
}

class CameraService:
    def __init__(self):
        self.detection_ws_url = settings.DETECTION_WEBSOCKET_URL
        self.server_http_url = settings.SERVER_HTTP_URL
        self.font = self._load_font()
        self.latest_detections = []
        self.latest_frame_to_send = None
        self.latest_frame_received = None # For decoupling
        self.running = False # Flag for detection_worker
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.selection_window_name = "Select Camera Source & Settings"
        self.stream_window_name = "Object Detection Stream"
        self.active_stream_task = None
        self.exit_service_flag = False 
        
        self.draw_masks = True
        self.draw_boxes = True
        
        self.languages = ["RUSSIAN", "ENGLISH", "KAZAKH"]
        self.current_lang_idx = 0

    def _load_font(self):
        try:
            return ImageFont.truetype(settings.VISUALIZATION_FONT_PATH, settings.VISUALIZATION_FONT_SIZE)
        except IOError:
            logger.warning(f"{settings.VISUALIZATION_FONT_PATH} not found, using default font. Russian characters might not display correctly.")
            return ImageFont.load_default()

    def _get_ui_text(self, key):
        lang = self.languages[self.current_lang_idx]
        return UI_TRANSLATIONS.get(lang, UI_TRANSLATIONS["ENGLISH"]).get(key, key)

    async def _fetch_available_cameras(self):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.server_http_url}/cameras")
                response.raise_for_status()
                return response.json()
        except (httpx.RequestError, json.JSONDecodeError) as e:
            logger.error(f"Error fetching camera list: {e}. Falling back to local webcam.")
            return [{"id": "local_0", "name": "Local Webcam (Fallback)", "type": "local"}]
            
    async def _update_language(self):
        """Notifies the backend about the language change."""
        lang = self.languages[self.current_lang_idx]
        logger.info(f"Switching language to: {lang}")
        try:
            async with httpx.AsyncClient() as client:
                await client.post(f"{self.server_http_url}/set_language", json={"language": lang})
        except Exception as e:
            logger.error(f"Failed to set language: {e}")

    def _select_file(self):
        """Opens file dialog to select image or video."""
        try:
            root = tk.Tk()
            root.withdraw() # Hide main window
            root.attributes('-topmost', True) # Bring to front
            root.focus_force() # Force focus
            
            file_path = filedialog.askopenfilename(
                title=self._get_ui_text("analyse_file"),
                filetypes=[
                    ("Image/Video", "*.jpg;*.jpeg;*.png;*.bmp;*.mp4;*.avi;*.mov;*.mkv"),
                    ("All files", "*.*")
                ]
            )
            root.destroy()
            return file_path
        except Exception as e:
            logger.error(f"Error opening file dialog: {e}")
            return None

    async def run_gui_and_stream_manager(self):
        cv2.namedWindow(self.selection_window_name)
        
        last_fetch_time = 0
        available_cameras = []
        last_camera_list_str = ""

        while not self.exit_service_flag:
            current_time = asyncio.get_event_loop().time()
            
            if current_time - last_fetch_time > 2.0:
                fetched_cameras = await self._fetch_available_cameras()
                if not fetched_cameras:
                    fetched_cameras = [{"id": "local_0", "name": "Local Webcam (Fallback)", "type": "local"}]
                
                if str(fetched_cameras) != last_camera_list_str or True:
                    available_cameras = fetched_cameras
                    last_camera_list_str = str(available_cameras)

                    num_settings_lines = 6
                    display_height = max(300, len(available_cameras) * 30 + num_settings_lines * 25 + 160)
                    selection_image = np.zeros((display_height, 600, 3), dtype=np.uint8)
                    selection_image.fill(40)

                    pil_img = Image.fromarray(cv2.cvtColor(selection_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)

                    y_offset = 20
                    draw.text((20, y_offset), self._get_ui_text("settings"), font=self.font, fill=(255, 255, 0))
                    y_offset += 30
                    draw.text((40, y_offset), f"Model: {settings.DETECTOR_MODEL_PATH}", font=self.font, fill=(200, 200, 200))
                    y_offset += 25
                    draw.text((40, y_offset), f"Confidence: {settings.DETECTOR_CONF_THRESHOLD}", font=self.font, fill=(200, 200, 200))
                    y_offset += 25
                    draw.text((40, y_offset), f"Server URL: {settings.SERVER_HTTP_URL}", font=self.font, fill=(200, 200, 200))
                    y_offset += 25
                    draw.text((40, y_offset), f"Detection WS: {settings.DETECTION_WEBSOCKET_URL}", font=self.font, fill=(200, 200, 200))
                    y_offset += 25
                    
                    lang_label = self._get_ui_text("lang")
                    lang_name = self.languages[self.current_lang_idx]
                    draw.text((40, y_offset), f"{lang_label}: {lang_name} [L]", font=self.font, fill=(0, 255, 255))
                    y_offset += 40

                    draw.text((20, y_offset), self._get_ui_text("select_src"), font=self.font, fill=(255, 255, 255))
                    y_offset += 30
                    for i, cam in enumerate(available_cameras):
                        draw.text((40, y_offset), f"{i + 1}. {cam['name']}", font=self.font, fill=(200, 200, 200))
                        y_offset += 30
                    
                    # Add Analyse File Option
                    draw.text((40, y_offset), f"P. {self._get_ui_text('analyse_file')}", font=self.font, fill=(100, 255, 100))
                    y_offset += 30
                    
                    draw.text((40, y_offset + 10), f"Q. {self._get_ui_text('quit')}", font=self.font, fill=(255, 100, 100))

                    selection_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    cv2.imshow(self.selection_window_name, selection_image)
                
                last_fetch_time = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.exit_service_flag = True
                logger.info("Quit key pressed. Exiting CameraService.")
                break
            elif key == ord('l') or key == ord('L'):
                self.current_lang_idx = (self.current_lang_idx + 1) % len(self.languages)
                await self._update_language()
            elif key == ord('p') or key == ord('P'):
                # Stop any active stream first
                if self.active_stream_task:
                    self.active_stream_task.cancel()
                    try: await self.active_stream_task
                    except: pass
                    self.active_stream_task = None
                
                try: cv2.destroyWindow(self.stream_window_name)
                except: pass

                # Handle file selection
                file_path = await asyncio.get_event_loop().run_in_executor(self.executor, self._select_file)
                if file_path:
                    logger.info(f"Selected file: {file_path}")
                    source = {"type": "file", "path": file_path, "name": "File Analysis"}
                    
                    # Run stream and check exit status
                    should_quit = await self._run_stream_task(source)
                    
                    if should_quit:
                        self.exit_service_flag = True
                        break
                    
                    logger.info("Returning to menu.")
                    # Clear key buffer rigorously
                    for _ in range(10): cv2.waitKey(1)
                    await asyncio.sleep(0.5) 

            if ord('1') <= key <= ord('9'):
                idx = int(chr(key)) - 1
                if 0 <= idx < len(available_cameras):
                    selected_camera = available_cameras[idx]
                    logger.info(f"Selected camera: {selected_camera['name']}")
                    
                    if self.active_stream_task:
                        self.active_stream_task.cancel()
                        try: await self.active_stream_task
                        except: pass
                        self.active_stream_task = None
                    
                    try: cv2.destroyWindow(self.stream_window_name)
                    except: pass

                    should_quit = await self._run_stream_task(selected_camera)
                    if should_quit:
                        self.exit_service_flag = True
                        break
                    
            await asyncio.sleep(0.01)

        if self.active_stream_task:
            self.active_stream_task.cancel()
            try: await self.active_stream_task
            except: pass
        cv2.destroyAllWindows()
        logger.info("CameraService manager shut down.")

    async def _run_stream_task(self, source):
        """
        Runs the stream. Returns True if the application should quit (user closed window),
        False if just returning to menu (user pressed Q).
        """
        video_capture = None
        stream_ws = None
        detection_ws = None
        detection_task = None
        receiver_task = None
        
        self.latest_frame_to_send = None
        self.latest_frame_received = None
        self.latest_detections = []
        self.running = True
        is_image_file = False
        static_image = None
        
        quit_app = False # Return value

        async def frame_receiver_loop():
            nonlocal video_capture, stream_ws, static_image
            logger.info("Frame receiver loop started.")
            loop = asyncio.get_event_loop()
            
            while self.running:
                try:
                    frame = None
                    
                    if source['type'] == 'file' and is_image_file:
                        if static_image is None:
                            try:
                                static_image = await loop.run_in_executor(self.executor, cv2.imread, source['path'])
                                if static_image is None:
                                    logger.error(f"Failed to read image file: {source['path']}")
                                    break
                            except Exception as e:
                                logger.error(f"Exception reading image: {e}")
                                break
                        
                        frame = static_image
                        await asyncio.sleep(0.1)
                        
                    elif video_capture:
                        ret, frame = await loop.run_in_executor(self.executor, video_capture.read)
                        if not ret:
                            if source['type'] == 'file':
                                await loop.run_in_executor(self.executor, video_capture.set, cv2.CAP_PROP_POS_FRAMES, 0)
                                continue
                            else:
                                logger.warning("End of stream/file.")
                                break 
                        
                        if source['type'] == 'local':
                            frame = cv2.flip(frame, 1)
                            
                    elif stream_ws:
                        try:
                            data = await asyncio.wait_for(stream_ws.recv(), timeout=1.0)
                            np_arr = np.frombuffer(data, np.uint8)
                            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        except asyncio.TimeoutError:
                            continue 
                    
                    if frame is not None:
                        self.latest_frame_received = frame
                        self.latest_frame_to_send = frame.copy() 
                    else:
                        await asyncio.sleep(0.01)
                        
                except Exception as e:
                    logger.error(f"Error in frame receiver: {e}")
                    await asyncio.sleep(1)
            logger.info("Frame receiver loop ended.")

        try:
            # 1. Connect Detection
            logger.info(f"Connecting to detection WebSocket: {self.detection_ws_url}")
            try:
                detection_ws = await websockets.connect(self.detection_ws_url, ping_interval=10, ping_timeout=10)
                detection_task = asyncio.create_task(self.detection_worker(detection_ws))
            except Exception as e:
                logger.error(f"Failed to connect to detection WebSocket: {e}")

            # 2. Connect Source
            if source['type'] == 'local':
                def open_cam():
                    cap = cv2.VideoCapture(settings.CAMERA_DEFAULT_ID)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_RESOLUTION_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_RESOLUTION_HEIGHT)
                    cap.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)
                    return cap
                video_capture = await asyncio.get_event_loop().run_in_executor(self.executor, open_cam)
                if not video_capture.isOpened():
                    return False
            elif source['type'] == 'remote_websocket':
                stream_ws = await websockets.connect(source['stream_ws_url'], ping_interval=10, ping_timeout=10)
            elif source['type'] == 'file':
                path = source['path']
                ext = os.path.splitext(path)[1].lower()
                is_image_file = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
                
                logger.info(f"Opening file: {path} (Image: {is_image_file})")
                
                if not is_image_file:
                    video_capture = cv2.VideoCapture(path)
                    if not video_capture.isOpened():
                        logger.error(f"Could not open file: {path}")
                        return False

            # 3. Start Receiver Background Task
            receiver_task = asyncio.create_task(frame_receiver_loop())

            # 4. Main UI/Display Loop (runs at ~30 FPS)
            cv2.namedWindow(self.stream_window_name, cv2.WINDOW_NORMAL) 
            
            # Initial resize for file analysis
            if source['type'] == 'file':
                for _ in range(10):
                    if self.latest_frame_received is not None:
                        h, w = self.latest_frame_received.shape[:2]
                        if w > 1280 or h > 720:
                            scale = min(1280/w, 720/h)
                            cv2.resizeWindow(self.stream_window_name, int(w*scale), int(h*scale))
                        else:
                            cv2.resizeWindow(self.stream_window_name, w, h)
                        break
                    await asyncio.sleep(0.1)

            while self.running:
                # Check for window close or Q
                try:
                    if cv2.getWindowProperty(self.stream_window_name, cv2.WND_PROP_VISIBLE) < 1:
                        # User closed window via X -> Quit App
                        quit_app = True
                        break
                except:
                    quit_app = True
                    break
                
                key = cv2.waitKey(30) & 0xFF 
                
                if key == ord('q'):
                    # User pressed Q -> Return to Menu
                    quit_app = False
                    break
                elif key == ord('m') or key == ord('M'):
                    self.draw_masks = not self.draw_masks
                    logger.info(f"Toggled Masks: {self.draw_masks}")
                elif key == ord('b') or key == ord('B'):
                    self.draw_boxes = not self.draw_boxes
                    logger.info(f"Toggled Boxes: {self.draw_boxes}")
                elif key == ord('l') or key == ord('L'):
                    self.current_lang_idx = (self.current_lang_idx + 1) % len(self.languages)
                    await self._update_language()

                # Display Frame if available
                if self.latest_frame_received is not None:
                    display_frame_copy = self.latest_frame_received.copy()
                    self.display_frame(display_frame_copy, self.latest_detections, self.stream_window_name)
                
                await asyncio.sleep(0.001)

        except Exception as e:
            logger.error(f"Error in stream task: {e}", exc_info=True)
        finally:
            self.running = False
            if receiver_task:
                receiver_task.cancel()
                try: await receiver_task
                except: pass
            
            if detection_task:
                detection_task.cancel()
                try: await detection_task
                except: pass
            
            if detection_ws: await detection_ws.close()
            if video_capture: video_capture.release()
            if stream_ws: await stream_ws.close()
            try: cv2.destroyWindow(self.stream_window_name)
            except: pass
            logger.info("Stream task finished cleanup.")
            
        return quit_app

    async def detection_worker(self, ws):
        logger.info("Detection worker started.")
        while self.running:
            try:
                # We use self.latest_frame_to_send which is updated by receiver loop
                if self.latest_frame_to_send is not None:
                    frame = self.latest_frame_to_send
                    h, w = frame.shape[:2]
                    scale = settings.DETECTOR_IMG_SIZE / w if w > settings.DETECTOR_IMG_SIZE else 1.0
                    if scale < 1.0:
                        small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    else:
                        small_frame = frame

                    _, buffer = cv2.imencode('.jpg', small_frame)
                    
                    await asyncio.wait_for(ws.send(buffer.tobytes()), timeout=5.0)
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(response)
                    self.latest_detections = data.get('boxes', [])
                else:
                    await asyncio.sleep(0.01)
            except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError):
                logger.warning("Detection WebSocket connection lost. Worker stopping.")
                break
            except Exception as e:
                logger.error(f"Error in detection worker: {e}")
                await asyncio.sleep(0.5)
        logger.info("Detection worker stopped.")

    def get_color_for_object(self, russian_name):
        english_name = None
        for en, ru in RUSSIAN_NAMES.items():
            if ru == russian_name:
                english_name = en
                break
        
        if not english_name:
            english_name = russian_name

        if english_name in HIGH_PRIORITY_OBJECTS: return (0, 0, 255) # Red (BGR)
        elif english_name in MEDIUM_PRIORITY_OBJECTS: return (0, 255, 255) # Yellow
        else: return (0, 255, 0) # Green

    def display_frame(self, frame, detections, window_name):
        h, w, _ = frame.shape
        overlay = frame.copy()
        alpha = settings.VISUALIZATION_ALPHA

        # 1. Draw Masks
        if self.draw_masks:
            for det in detections:
                if det.get('mask_points'):
                    points = det['mask_points']
                    if points:
                        scaled_points = (np.array(points) * np.array([w, h])).astype(np.int32)
                        color = self.get_color_for_object(det.get('name', ''))
                        cv2.fillPoly(overlay, [scaled_points], color)
        
        # 2. Blend Overlay
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # 3. Draw Boxes & Text (using PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

        if self.draw_boxes:
            for det in detections:
                if all(k in det for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                    x1, y1 = int(det['xmin'] * w), int(det['ymin'] * h)
                    x2, y2 = int(det['xmax'] * w), int(det['ymax'] * h)
                    
                    color_bgr = self.get_color_for_object(det.get('name', ''))
                    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

                    # Box
                    draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)

                    # Label
                    label = f"{det.get('name', 'Unknown')}"
                    if det.get('distance_cm') is not None:
                        label += f" ({det['distance_cm']:.0f} cm)"
                    
                    bbox = draw.textbbox((0, 0), label, font=self.font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
                    
                    draw.rectangle((x1, text_y - text_height - 5, x1 + text_width + 10, text_y + 5), fill=color_rgb)
                    draw.text((x1 + 5, text_y - text_height - 5), label, font=self.font, fill=(255, 255, 255))

        # 4. Instructions HUD
        mask_status = self._get_ui_text("on") if self.draw_masks else self._get_ui_text("off")
        box_status = self._get_ui_text("on") if self.draw_boxes else self._get_ui_text("off")
        lang_name = self.languages[self.current_lang_idx]
        
        instructions = [
            f"[M] {self._get_ui_text('masks')}: {mask_status}",
            f"[B] {self._get_ui_text('boxes')}: {box_status}",
            f"[L] {self._get_ui_text('lang')}: {lang_name}",
            f"[Q] {self._get_ui_text('quit')}"
        ]
        
        y_text = 10
        for instr in instructions:
            draw.text((10, y_text), instr, font=self.font, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0,0,0))
            y_text += 25

        final_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, final_frame)

async def main():
    camera_service = CameraService()
    await camera_service.run_gui_and_stream_manager()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
