import asyncio
import cv2
import websockets
import json
import numpy as np
import colorsys
from PIL import Image, ImageDraw, ImageFont
import httpx
import logging
from concurrent.futures import ThreadPoolExecutor

from app.domain.priorities import HIGH_PRIORITY_OBJECTS, MEDIUM_PRIORITY_OBJECTS
from app.services.detector import RUSSIAN_NAMES
from app.config import settings

# Configure logging for this service
logger = logging.getLogger(__name__)

class CameraService:
    def __init__(self):
        self.detection_ws_url = settings.DETECTION_WEBSOCKET_URL
        self.server_http_url = settings.SERVER_HTTP_URL
        self.font = self._load_font()
        self.latest_detections = []
        self.latest_frame_to_send = None
        self.running = False # Flag for detection_worker
        self.executor = ThreadPoolExecutor(max_workers=1)

        self.selection_window_name = "Select Camera Source & Settings"
        self.stream_window_name = "Object Detection Stream"
        self.active_stream_task = None
        self.exit_service_flag = False # Flag to exit the entire CameraService

    def _load_font(self):
        try:
            return ImageFont.truetype(settings.VISUALIZATION_FONT_PATH, settings.VISUALIZATION_FONT_SIZE)
        except IOError:
            logger.warning(f"{settings.VISUALIZATION_FONT_PATH} not found, using default font. Russian characters might not display correctly.")
            return ImageFont.load_default()

    async def _fetch_available_cameras(self):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.server_http_url}/cameras")
                response.raise_for_status()
                return response.json()
        except (httpx.RequestError, json.JSONDecodeError) as e:
            logger.error(f"Error fetching camera list: {e}. Falling back to local webcam.")
            return [{"id": "local_0", "name": "Local Webcam (Fallback)", "type": "local"}]

    async def run_gui_and_stream_manager(self):
        cv2.namedWindow(self.selection_window_name)
        
        last_fetch_time = 0
        available_cameras = []
        last_camera_list_str = ""

        while not self.exit_service_flag:
            current_time = asyncio.get_event_loop().time()
            
            # Refresh list every 2 seconds
            if current_time - last_fetch_time > 2.0:
                fetched_cameras = await self._fetch_available_cameras()
                if not fetched_cameras:
                    fetched_cameras = [{"id": "local_0", "name": "Local Webcam (Fallback)", "type": "local"}]
                
                # Only redraw if the list has changed
                if str(fetched_cameras) != last_camera_list_str:
                    logger.info(f"Camera list updated: {fetched_cameras}")
                    available_cameras = fetched_cameras
                    last_camera_list_str = str(available_cameras)

                    num_settings_lines = 5
                    display_height = max(300, len(available_cameras) * 30 + num_settings_lines * 25 + 120)
                    selection_image = np.zeros((display_height, 600, 3), dtype=np.uint8)
                    selection_image.fill(40)

                    pil_img = Image.fromarray(cv2.cvtColor(selection_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)

                    y_offset = 20
                    draw.text((20, y_offset), "KOZAINEK Settings:", font=self.font, fill=(255, 255, 0))
                    y_offset += 30
                    draw.text((40, y_offset), f"Model: {settings.DETECTOR_MODEL_PATH}", font=self.font, fill=(200, 200, 200))
                    y_offset += 25
                    draw.text((40, y_offset), f"Confidence: {settings.DETECTOR_CONF_THRESHOLD}", font=self.font, fill=(200, 200, 200))
                    y_offset += 25
                    draw.text((40, y_offset), f"Server URL: {settings.SERVER_HTTP_URL}", font=self.font, fill=(200, 200, 200))
                    y_offset += 25
                    draw.text((40, y_offset), f"Detection WS: {settings.DETECTION_WEBSOCKET_URL}", font=self.font, fill=(200, 200, 200))
                    y_offset += 40

                    draw.text((20, y_offset), "Select a Camera Source:", font=self.font, fill=(255, 255, 255))
                    y_offset += 30
                    for i, cam in enumerate(available_cameras):
                        draw.text((40, y_offset), f"{i + 1}. {cam['name']}", font=self.font, fill=(200, 200, 200))
                        y_offset += 30
                    draw.text((40, y_offset + 10), "Q. Quit Service", font=self.font, fill=(255, 100, 100))

                    selection_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    cv2.imshow(self.selection_window_name, selection_image)
                
                last_fetch_time = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.exit_service_flag = True
                logger.info("Quit key pressed. Exiting CameraService.")
                break

            if ord('1') <= key <= ord('9'):
                idx = int(chr(key)) - 1
                if 0 <= idx < len(available_cameras):
                    selected_camera = available_cameras[idx]
                    logger.info(f"Selected camera: {selected_camera['name']}")
                    
                    # Cancel any existing stream task
                    if self.active_stream_task:
                        logger.info("Cancelling previous stream task.")
                        self.active_stream_task.cancel()
                        try:
                            await self.active_stream_task
                        except asyncio.CancelledError:
                            pass
                        # Removed: cv2.destroyWindow(self.stream_window_name) # This is now handled by _run_stream_task's finally block
                    
                    # Start new stream task
                    self.active_stream_task = asyncio.create_task(self._run_stream_task(selected_camera))
                    
            await asyncio.sleep(0.01) # Yield control

        # Final cleanup when the manager loop exits
        if self.active_stream_task:
            self.active_stream_task.cancel()
            try:
                await self.active_stream_task
            except asyncio.CancelledError:
                pass
        cv2.destroyAllWindows() # Destroy all remaining OpenCV windows
        logger.info("CameraService manager shut down.")

    async def detection_worker(self, ws):
        """Background task to handle sending frames and receiving detections."""
        logger.info("Detection worker started.")
        while self.running:
            if self.latest_frame_to_send is not None:
                frame = self.latest_frame_to_send
                
                h, w = frame.shape[:2]
                scale = settings.DETECTOR_IMG_SIZE / w if w > settings.DETECTOR_IMG_SIZE else 1.0
                if scale < 1.0:
                    small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    small_frame = frame

                _, buffer = cv2.imencode('.jpg', small_frame)
                
                try:
                    await ws.send(buffer.tobytes())
                    response = await ws.recv()
                    data = json.loads(response)
                    self.latest_detections = data.get('boxes', [])
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Detection worker: Connection was closed during operation.")
                    break
                except asyncio.CancelledError:
                    logger.info("Detection worker: Task was cancelled.")
                    break
                except Exception as e:
                    if self.running:
                        logger.error(f"An unexpected error occurred in detection worker: {e}")
                    await asyncio.sleep(0.1)
            else:
                await asyncio.sleep(0.01)
        logger.info("Detection worker finished.")

    async def _run_stream_task(self, source):
        video_capture = None
        stream_ws = None
        detection_ws = None
        self.running = True # Signal detection_worker to run
        self.latest_detections = []
        self.latest_frame_to_send = None
        detection_task = None

        try:
            logger.info(f"Connecting to detection WebSocket: {self.detection_ws_url}")
            detection_ws = await websockets.connect(self.detection_ws_url)
            
            detection_task = asyncio.create_task(self.detection_worker(detection_ws))

            if source['type'] == 'local':
                video_capture = cv2.VideoCapture(settings.CAMERA_DEFAULT_ID)
                video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, settings.CAMERA_RESOLUTION_WIDTH)
                video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.CAMERA_RESOLUTION_HEIGHT)
                video_capture.set(cv2.CAP_PROP_FPS, settings.CAMERA_FPS)

                if not video_capture.isOpened():
                    logger.error("Could not open local video stream.")
                    return
                logger.info("Local webcam opened.")
            elif source['type'] == 'remote_websocket':
                stream_ws_url = source.get('stream_ws_url')
                if not stream_ws_url:
                    logger.error("Remote source has no 'stream_ws_url'.")
                    return
                logger.info(f"Connecting to remote stream: {stream_ws_url}")
                stream_ws = await websockets.connect(stream_ws_url)

            loop = asyncio.get_event_loop()
            cv2.namedWindow(self.stream_window_name) # Create stream display window

            while True: # Loop until break or cancelled
                frame = None
                if video_capture:
                    ret, frame = await loop.run_in_executor(self.executor, video_capture.read)
                    if not ret:
                        logger.warning("Failed to read frame from local camera.")
                        break
                    frame = cv2.flip(frame, 1)
                elif stream_ws:
                    try:
                        data = await stream_ws.recv()
                        np_arr = np.frombuffer(data, np.uint8)
                        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Remote stream connection closed.")
                        break
                
                if frame is None: continue

                self.latest_frame_to_send = frame.copy()
                self.display_frame(frame, self.latest_detections, self.stream_window_name)

                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(self.stream_window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("User closed the stream window or pressed 'q'.")
                    break
                
                await asyncio.sleep(0.001) # Yield control
        
        except asyncio.CancelledError:
            logger.info("Stream task was cancelled.")
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in stream task: {e}", exc_info=True)
        finally:
            self.running = False # Signal detection_worker to stop
            if detection_task:
                detection_task.cancel()
                try:
                    await detection_task
                except asyncio.CancelledError:
                    pass

            logger.info("Cleaning up stream task resources...")
            if video_capture: video_capture.release()
            if stream_ws: await stream_ws.close()
            if detection_ws: await detection_ws.close()
            try:
                cv2.destroyWindow(self.stream_window_name) # Close stream window
            except cv2.error:
                logger.warning(f"Window {self.stream_window_name} already destroyed.")
            self.active_stream_task = None # Clear reference to this task
            logger.info("Stream task finished.")

    def get_color_for_object(self, russian_name):
        english_name = next((en for en, ru in RUSSIAN_NAMES.items() if ru == russian_name), None)
        if english_name in HIGH_PRIORITY_OBJECTS: return (0, 0, 255)
        elif english_name in MEDIUM_PRIORITY_OBJECTS: return (0, 255, 255)
        else: return (0, 255, 0)

    def display_frame(self, frame, detections, window_name):
        h, w, _ = frame.shape
        overlay = frame.copy()
        alpha = settings.VISUALIZATION_ALPHA

        for det in detections:
            color = self.get_color_for_object(det.get('name', ''))
            if det.get('mask_points'):
                scaled_points = (np.array(det['mask_points']) * np.array([w, h])).astype(np.int32)
                cv2.fillPoly(overlay, [scaled_points], color)
            
            if all(k in det for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                x1, y1 = int(det['xmin'] * w), int(det['ymin'] * h)
                x2, y2 = int(det['xmax'] * w), int(det['ymax'] * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        for det in detections:
            if all(k in det for k in ['xmin', 'ymin']):
                x1, y1 = int(det['xmin'] * w), int(det['ymin'] * h)
                label = f"{det.get('name', '')}"
                if det.get('distance_cm') is not None:
                    label += f" ({det['distance_cm']:.0f} cm)"
                
                bbox = draw.textbbox((0, 0), label, font=self.font)
                text_height = bbox[3] - bbox[1]
                text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
                
                color_bgr = self.get_color_for_object(det.get('name', ''))
                draw.rectangle((x1, text_y - text_height - 5, x1 + bbox[2] - bbox[0] + 10, text_y + 5), fill=(color_bgr[2], color_bgr[1], color_bgr[0]))
                draw.text((x1 + 5, text_y - text_height - 5), label, font=self.font, fill=(255, 255, 255))

        cv2.imshow(window_name, cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))

async def main():
    camera_service = CameraService()
    await camera_service.run_gui_and_stream_manager()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
