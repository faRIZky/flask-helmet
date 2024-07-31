from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
from datetime import datetime
import logging
import base64
import time

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load YOLOv8 model
model_path = 'best_helmet_addy_yolov8n.pt'
model = YOLO(model_path)

# Global list to store detection results
detection_results = []

# Function to process video stream
def process_video(video_source, camera_name):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logging.error(f"Cannot open camera with source {video_source}")
        return

    last_detection_time = 0
    detection_interval = 5  # seconds

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logging.error("Can't receive frame (stream end?). Exiting ...")
            break

        # Define ROIs based on camera
        '''
        kamera 1: 2/3 dari kiri ke kanan dan 1/2 dari atas ke bawah
        kamera 2: 2/3 dari kanan ke kiri dan 1/2 dari bawah ke atas
        '''
        height, width, _ = frame.shape
        if camera_name == "Laptop Kamera":
            x1, y1, x2, y2 = 0, 0, int(width * 2 / 3), int(height / 2)
        else:  # "Webcam Logitech"
            x1, y1, x2, y2 = int(width / 3), int(height / 2), width, height

        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        current_time = time.time()
        if current_time - last_detection_time >= detection_interval:
            last_detection_time = current_time

            try:
                # Crop the frame to the ROI
                roi_frame = frame[y1:y2, x1:x2]

                # Object detection
                results = model(roi_frame)
                for result in results:
                    for box in result.boxes:
                        conf = box.conf[0].cpu().numpy()
                        cls = box.cls[0].cpu().numpy()
                        if conf > 0.5:  # Display detections with confidence > 0.5
                            if model.names[int(cls)] == "NO-Hardhat":
                                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                x1_box, y1_box, x2_box, y2_box = box.xyxy[0].cpu().numpy()
                                x1_box, x2_box = x1 + x1_box, x1 + x2_box
                                y1_box, y2_box = y1 + y1_box, y1 + y2_box
                                crop_img = frame[int(y1_box):int(y2_box), int(x1_box):int(x2_box)]
                                ret, buffer = cv2.imencode('.jpg', crop_img)
                                crop_img_encoded = base64.b64encode(buffer).decode('utf-8')
                                detection_results.append((crop_img_encoded, timestamp, camera_name))
                                if len(detection_results) > 10:  # Limit the list to the last 10 detections
                                    detection_results.pop(0)
                            x1_box, y1_box, x2_box, y2_box = box.xyxy[0].cpu().numpy()
                            x1_box, x2_box = x1 + x1_box, x1 + x2_box
                            y1_box, y2_box = y1 + y1_box, y1 + y2_box
                            logging.debug(f"Drawing box: ({x1_box}, {y1_box}), ({x2_box}, {y2_box})")
                            cv2.rectangle(frame, (int(x1_box), int(y1_box)), (int(x2_box), int(y2_box)), (0, 255, 0), 2)
                            cv2.putText(frame, f'{model.names[int(cls)]} {conf:.2f}', (int(x1_box), int(y1_box) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except Exception as e:
                logging.error(f"Error during object detection: {e}")

        # Convert frame to JPEG format for live video feed
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed_laptop')
def video_feed_laptop():
    return Response(process_video(0, "Laptop Kamera"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_logitech')
def video_feed_logitech():
    return Response(process_video(1, "Webcam Logitech"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def detections():
    return jsonify(detection_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
