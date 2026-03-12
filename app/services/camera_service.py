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

# Configure logging for this service
logger = logging.getLogger(__name__)

class CameraService:
    def __init__(self, detection_websocket_url="ws://localhost:8000/ws", server_http_url="http://localhost:8000"):
        self.detection_ws_url = detection_websocket_url
        self.server_http_url = server_http_url
        self.font = self._load_font()
        self.latest_detections = []
        self.latest_frame_to_send = None
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=1)

    def _load_font(self):
        try:
            return ImageFont.truetype("arial.ttf", 20)
        except IOError:
            logger.warning("arial.ttf not found, using default font. Russian characters might not display correctly.")
            return ImageFont.load_default()

    async def select_camera_source(self):
        selection_window_name = "Select Camera Source"
        cv2.namedWindow(selection_window_name)

        last_fetch_time = 0
        available_cameras = []
        last_camera_list_str = ""

        while True:
            current_time = asyncio.get_event_loop().time()
            
            # Refresh list every 2 seconds
            if current_time - last_fetch_time > 2.0:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{self.server_http_url}/cameras")
                        response.raise_for_status()
                        fetched_cameras = response.json()
                except (httpx.RequestError, json.JSONDecodeError):
                    fetched_cameras = [{"id": "local_0", "name": "Local Webcam (Fallback)", "type": "local"}]

                if not fetched_cameras:
                    fetched_cameras = [{"id": "local_0", "name": "Local Webcam (Fallback)", "type": "local"}]
                
                # Only redraw if the list has changed
                if str(fetched_cameras) != last_camera_list_str:
                    logger.info(f"Camera list updated: {fetched_cameras}")
                    available_cameras = fetched_cameras
                    last_camera_list_str = str(available_cameras)

                    display_height = max(200, len(available_cameras) * 30 + 80)
                    selection_image = np.zeros((display_height, 600, 3), dtype=np.uint8)
                    selection_image.fill(40)

                    pil_img = Image.fromarray(cv2.cvtColor(selection_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pil_img)

                    draw.text((20, 20), "Select a Camera Source (Auto-refreshing):", font=self.font, fill=(255, 255, 255))
                    y_offset = 60
                    for i, cam in enumerate(available_cameras):
                        draw.text((40, y_offset), f"{i + 1}. {cam['name']}", font=self.font, fill=(200, 200, 200))
                        y_offset += 30
                    draw.text((40, y_offset + 10), "Q. Quit", font=self.font, fill=(255, 100, 100))

                    selection_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    cv2.imshow(selection_window_name, selection_image)
                
                last_fetch_time = current_time

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow(selection_window_name)
                return None

            if ord('1') <= key <= ord('9'):
                idx = int(chr(key)) - 1
                if 0 <= idx < len(available_cameras):
                    selected_camera = available_cameras[idx]
                    cv2.destroyWindow(selection_window_name)
                    return selected_camera

            await asyncio.sleep(0.01)

    async def run_camera_loop(self):
        while True:
            selected_source = await self.select_camera_source()
            if selected_source is None:
                logger.info("Exiting camera service.")
                break

            await self.process_stream(selected_source)
            logger.info("Stream ended. Returning to selection screen.")

    async def detection_worker(self, ws):
        """Background task to handle sending frames and receiving detections."""
        logger.info("Detection worker started.")
        while self.running:
            if self.latest_frame_to_send is not None:
                frame = self.latest_frame_to_send
                
                # Resize frame to reduce bandwidth
                h, w = frame.shape[:2]
                scale = 640 / w if w > 640 else 1.0
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
                    break # Exit loop cleanly
                except asyncio.CancelledError:
                    logger.info("Detection worker: Task was cancelled.")
                    break # Exit loop cleanly
                except Exception as e:
                    if self.running: # Only log error if we are not in the process of shutting down
                        logger.error(f"An unexpected error occurred in detection worker: {e}")
                    await asyncio.sleep(0.1) # Backoff slightly on other errors
            else:
                await asyncio.sleep(0.01)
        logger.info("Detection worker finished.")

    async def process_stream(self, source):
        video_capture = None
        stream_ws = None
        detection_ws = None
        window_name = "Object Detection"
        self.running = True
        self.latest_detections = []
        self.latest_frame_to_send = None
        detection_task = None

        try:
            logger.info(f"Connecting to detection WebSocket: {self.detection_ws_url}")
            detection_ws = await websockets.connect(self.detection_ws_url)
            
            # Start the background detection task
            detection_task = asyncio.create_task(self.detection_worker(detection_ws))

            if source['type'] == 'local':
                video_capture = cv2.VideoCapture(0)
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

            while self.running:
                frame = None
                if video_capture:
                    # Run blocking read in thread pool to avoid freezing async loop
                    ret, frame = await loop.run_in_executor(self.executor, video_capture.read)
                    if not ret:
                        logger.warning("Failed to read frame from local camera.")
                        break
                    # Flip the local webcam frame horizontally
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

                # Update the frame available for the detection worker
                self.latest_frame_to_send = frame.copy()
                
                # Draw the latest known detections on the current frame
                self.display_frame(frame, self.latest_detections, window_name)

                # Check if user closed the window or pressed 'q'
                if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    logger.info("User closed the stream window or pressed 'q'.")
                    self.running = False # Signal all loops to stop
                    break
                
                # Small sleep to allow other tasks (like detection_worker) to run
                await asyncio.sleep(0.001)
        
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred in process_stream: {e}", exc_info=True)
        finally:
            self.running = False # Ensure running flag is false
            if detection_task:
                detection_task.cancel()
                try:
                    await detection_task
                except asyncio.CancelledError:
                    pass

            logger.info("Cleaning up stream resources...")
            if video_capture: video_capture.release()
            if stream_ws: await stream_ws.close()
            if detection_ws: await detection_ws.close()
            cv2.destroyAllWindows()

    def get_color_for_object(self, russian_name):
        english_name = next((en for en, ru in RUSSIAN_NAMES.items() if ru == russian_name), None)
        if english_name in HIGH_PRIORITY_OBJECTS: return (0, 0, 255)
        elif english_name in MEDIUM_PRIORITY_OBJECTS: return (0, 255, 255)
        else: return (0, 255, 0)

    def display_frame(self, frame, detections, window_name):
        h, w, _ = frame.shape
        overlay = frame.copy()
        alpha = 0.3

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
    await camera_service.run_camera_loop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
