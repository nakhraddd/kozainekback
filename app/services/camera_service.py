import asyncio
import cv2
import websockets
import json
import numpy as np
import colorsys
from PIL import Image, ImageDraw, ImageFont
import httpx

from app.domain.priorities import HIGH_PRIORITY_OBJECTS, MEDIUM_PRIORITY_OBJECTS
from app.services.detector import RUSSIAN_NAMES

class CameraService:
    def __init__(self, detection_websocket_url="ws://localhost:8000/ws", server_http_url="http://localhost:8000"):
        self.detection_ws_url = detection_websocket_url
        self.server_http_url = server_http_url
        self.font = self._load_font()

    def _load_font(self):
        try:
            return ImageFont.truetype("arial.ttf", 20)
        except IOError:
            print("Warning: arial.ttf not found, using default font.")
            return ImageFont.load_default()

    async def select_camera_source(self):
        selection_window_name = "Select Camera Source"
        cv2.namedWindow(selection_window_name)

        last_fetch_time = 0
        available_cameras = []

        while True:
            current_time = asyncio.get_event_loop().time()

            if current_time - last_fetch_time > 1.0:
                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{self.server_http_url}/cameras")
                        response.raise_for_status()
                        available_cameras = response.json()
                except (httpx.RequestError, json.JSONDecodeError):
                    available_cameras = [{"id": "local_0", "name": "Local Webcam (Fallback)", "type": "local"}]

                if not available_cameras:
                    available_cameras = [{"id": "local_0", "name": "Local Webcam (Fallback)", "type": "local"}]

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
                print("Exiting camera service.")
                break

            await self.process_stream(selected_source)
            print("Stream ended. Returning to selection screen.")

    async def process_stream(self, source):
        video_capture = None
        stream_ws = None
        detection_ws = None

        try:
            # Connect to the detection websocket first
            detection_ws = await websockets.connect(self.detection_ws_url)
            print(f"Connected to detection WebSocket: {self.detection_ws_url}")

            # Connect to the frame source (local or remote)
            if source['type'] == 'local':
                video_capture = cv2.VideoCapture(0)
                if not video_capture.isOpened():
                    print("Error: Could not open local video stream.")
                    return
            elif source['type'] == 'remote_websocket':
                stream_ws_url = source.get('stream_ws_url')
                if not stream_ws_url:
                    print("Error: Remote source has no 'stream_ws_url'.")
                    return
                stream_ws = await websockets.connect(stream_ws_url)
                print(f"Connected to remote stream: {stream_ws_url}")

            while True:
                frame = None
                if video_capture:
                    ret, frame = video_capture.read()
                    if not ret: break
                elif stream_ws:
                    data = await stream_ws.recv()
                    np_arr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None: continue

                _, buffer = cv2.imencode('.jpg', frame)
                await detection_ws.send(buffer.tobytes())
                
                response = await detection_ws.recv()
                detections = json.loads(response).get('boxes', [])
                
                self.display_frame(frame, detections)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except (websockets.exceptions.ConnectionClosed, httpx.RequestError) as e:
            print(f"Connection error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            if video_capture: video_capture.release()
            if stream_ws: await stream_ws.close()
            if detection_ws: await detection_ws.close()
            cv2.destroyAllWindows()

    def get_color_for_object(self, russian_name):
        english_name = next((en for en, ru in RUSSIAN_NAMES.items() if ru == russian_name), None)
        if english_name in HIGH_PRIORITY_OBJECTS: return (0, 0, 255)
        elif english_name in MEDIUM_PRIORITY_OBJECTS: return (0, 255, 255)
        else: return (0, 255, 0)

    def display_frame(self, frame, detections):
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

        cv2.imshow('Object Detection', cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))

async def main():
    camera_service = CameraService()
    await camera_service.run_camera_loop()

if __name__ == "__main__":
    asyncio.run(main())
