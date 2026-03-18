# KOZAINEK

Computer vision for blind and visually impaired people.

**Goal:** Create a real-time object recognition system with voice output in Russian, which helps safely navigate in space.

---

## 1. Functionality (Current State - MVP+)

The current implementation includes an enhanced MVP with the following features:

### 1.1. Object Detection
*   Uses the **YOLOv8** neural network (model `yolov8n-seg.pt`).
*   Recognizes **80+ types of objects**: people, vehicles, animals, furniture, and household items.
*   Works in real-time on CPU and GPU (supports NVIDIA CUDA).

### 1.2. Distance Analysis
*   Distance estimation based on object size in the frame.
*   Classification: "close", "at a medium distance", "far".
*   Uses reference sizes for 30+ object types.

### 1.3. Position Detection
*   Determining the object's position relative to the user:
    *   Horizontally: "left", "center", "right".
    *   Vertically: "top", "middle", "bottom".

### 1.4. Obstacle Detection
*   Detection of vertical lines (wall corners, doors, pillars) using Canny and Hough Transform algorithms.
*   Warning: "Attention! Possible obstacle ahead".

### 1.5. Voice Output
*   Speech synthesis in Russian (uses system Windows voice).
*   Asynchronous operation (does not block the video stream).
*   Announcing object type, position, and distance.

### 1.6. Priority System
*   **High priority (dangerous):** people, vehicles, dogs, traffic lights.
*   **Medium priority (obstacles):** chairs, tables, benches.
*   **Low priority:** other objects.

### 1.7. Announcement Manager
*   Spam protection: the same object is not announced more often than once every 4-6 seconds.
*   Automatic announcements every 6 seconds (blind mode).

### 1.8. Web Interface and Remote Access
*   Web server based on **FastAPI**.
*   Support for connecting remote cameras (e.g., from a smartphone) via WebSocket.
*   Tunneling via **ngrok** for internet access.

### 1.9. Visualization (for sighted assistants)
*   Displaying video with object bounding boxes:
    *   Red = close (dangerous).
    *   Yellow = medium distance.
    *   Green = far.
*   Displaying recognition confidence and segmentation masks.

---

## 2. Installation and Launch

### Requirements
*   Windows 10/11 (64-bit)
*   Python 3.10+
*   Webcam

### Instructions
  [download](https://github.com/nakhraddd/kozainekback/releases/download/latest/Kozainek-Windows.zip)

  Extract kozainek folder

  **Launch kozainek.exe:**

---

## 3. Usage

### For the Blind User
1.  Run the program.
2.  Wait for the voice greeting.
3.  Point the camera in front of you.
4.  The system will automatically announce found objects and obstacles every few seconds.

### For the Assistant / Developer
*   Local server available at: `http://localhost:8000`
*   Camera list: `http://localhost:8000/cameras`
*   Settings are located in `app/config.py`.

---

## 4. Tech Stack
*   **Language:** Python 3.11
*   **AI:** Ultralytics YOLOv8 (PyTorch)
*   **Backend:** FastAPI, Uvicorn, WebSockets
*   **Computer Vision:** OpenCV, NumPy
*   **Tunneling:** pyngrok

---

## 5. Privacy
*   All calculations are performed locally on your device.
*   Video stream and images are not sent to the cloud.
*   The system does not collect personal data.

---

**Developed by:** Darkhan Tastanov (2026)