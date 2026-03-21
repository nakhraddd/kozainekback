import os
import shutil
import requests
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
OUTPUT_NAME = "yolo.tflite"

def download_file(url, filename):
    if os.path.exists(filename):
        print(f"Model {filename} already exists.")
        return

    print(f"Downloading model from {url} to {filename}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        exit(1)

def main():
    # 1. Ensure the source model file exists
    download_file(MODEL_URL, MODEL_NAME)

    print(f"Loading YOLO model: {MODEL_NAME}...")
    model = YOLO(MODEL_NAME)

    print("Exporting model to TFLite format...")
    # Export returns the path to the exported file
    exported_path = model.export(format="tflite")
    
    print(f"Export finished. Output located at: {exported_path}")

    # The export creates a file named 'yolov8n.tflite' usually in the same directory
    # We want to rename it to 'yolo.tflite'
    
    # Check if the exported path is effectively what we want, or move it
    if exported_path and os.path.exists(exported_path):
        if os.path.basename(exported_path) != OUTPUT_NAME:
            print(f"Renaming {exported_path} to {OUTPUT_NAME}...")
            if os.path.exists(OUTPUT_NAME):
                os.remove(OUTPUT_NAME)
            shutil.move(exported_path, OUTPUT_NAME)
            print(f"Successfully created {OUTPUT_NAME}")
        else:
             print(f"File is already named {OUTPUT_NAME}")
    else:
        # Fallback search if return value isn't what we expect
        potential_output = MODEL_NAME.replace(".pt", ".tflite")
        if os.path.exists(potential_output):
             print(f"Renaming {potential_output} to {OUTPUT_NAME}...")
             if os.path.exists(OUTPUT_NAME):
                os.remove(OUTPUT_NAME)
             shutil.move(potential_output, OUTPUT_NAME)
             print(f"Successfully created {OUTPUT_NAME}")
        else:
            print(f"Could not locate the exported tflite file. Please check the output above.")

if __name__ == "__main__":
    main()
