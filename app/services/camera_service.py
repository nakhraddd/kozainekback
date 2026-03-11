import asyncio
import cv2
import websockets
import json
import numpy as np
import colorsys

from app.domain.priorities import HIGH_PRIORITY_OBJECTS, MEDIUM_PRIORITY_OBJECTS
from app.services.detector import RUSSIAN_NAMES # Import RUSSIAN_NAMES from detector.py

class CameraService:
    def __init__(self, websocket_url="ws://localhost:8000/ws"):
        self.video_capture = cv2.VideoCapture(0)
        self.websocket_url = websocket_url
        self.websocket = None
        # No need for self.colors anymore as colors are priority-based
        self.english_to_russian = {v: k for k, v in RUSSIAN_NAMES.items()} # Reverse mapping

    async def connect_to_websocket(self):
        self.websocket = await websockets.connect(self.websocket_url)

    async def start_camera(self):
        if not self.video_capture.isOpened():
            print("Error: Could not open video stream.")
            return

        await self.connect_to_websocket()

        while True:
            ret, frame = self.video_capture.read()
            if not ret:
                break

            # Encode frame to JPEG for sending over WebSocket
            _, buffer = cv2.imencode('.jpg', frame)
            await self.websocket.send(buffer.tobytes())

            # Receive detection data from WebSocket
            response = await self.websocket.recv()
            data = json.loads(response) # This will be a dictionary {"text": "...", "boxes": [...]}
            
            # Extract the 'boxes' list which contains the detection objects
            detections = data.get('boxes', [])

            self.display_frame(frame, detections)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.video_capture.release()
        cv2.destroyAllWindows()
        await self.websocket.close()

    def get_color_for_object(self, russian_name):
        english_name = self.english_to_russian.get(russian_name)

        if english_name in HIGH_PRIORITY_OBJECTS:
            return (0, 0, 255) # Red (BGR)
        elif english_name in MEDIUM_PRIORITY_OBJECTS:
            return (0, 255, 255) # Yellow (BGR)
        else:
            return (0, 255, 0) # Green (BGR) - Low priority or unknown

    def display_frame(self, frame, detections):
        h, w, _ = frame.shape
        overlay = frame.copy()
        alpha = 0.3 # Transparency factor for the mask

        for det in detections:
            name = det.get('name', 'Unknown') # This is the Russian name
            
            # The box coordinates are now normalized (xmin, ymin, xmax, ymax)
            xmin = det.get('xmin')
            ymin = det.get('ymin')
            xmax = det.get('xmax')
            ymax = det.get('ymax')
            
            mask_points = det.get('mask_points')
            distance = det.get('distance_cm') # The key is 'distance_cm'

            color = self.get_color_for_object(name)

            # Draw bounding box (converting normalized coords to pixel coords)
            if all(v is not None for v in [xmin, ymin, xmax, ymax]):
                x1, y1 = int(xmin * w), int(ymin * h)
                x2, y2 = int(xmax * w), int(ymax * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Display name and distance
                label = f"{name}"
                if distance is not None:
                    label += f" ({distance:.0f} cm)"
                
                # Adjust text position to be above the box
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = x1
                text_y = y1 - 10 if y1 - 10 > text_size[1] else y1 + text_size[1] + 10 # Avoid going off screen
                
                cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), 
                              (text_x + text_size[0] + 5, text_y + 5), color, -1)
                cv2.putText(frame, label, (text_x + 2, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Draw mask
            if mask_points and len(mask_points) > 0:
                # The mask_points variable is a single list of points for one polygon.
                # We need to convert it to a NumPy array of the correct shape and type.
                scaled_points = (np.array(mask_points) * np.array([w, h])).astype(np.int32)
                cv2.fillPoly(overlay, [scaled_points], color)

        # Blend the overlay with the original frame
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.imshow('Object Detection', frame)

async def main():
    camera_service = CameraService()
    await camera_service.start_camera()

if __name__ == "__main__":
    asyncio.run(main())
