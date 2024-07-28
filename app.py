from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_segments'
FULL_VIDEO_OUTPUT = 'full_detection_output.mp4'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['FULL_VIDEO_OUTPUT'] = FULL_VIDEO_OUTPUT

# Ensure the upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            output_dir = process_video(file_path)
            return redirect(url_for('show_results', output_dir=output_dir))
    return render_template('upload.html')

@app.route('/results')
def show_results():
    output_dir = request.args.get('output_dir')
    segments = [f for f in os.listdir(output_dir) if f.endswith('.mp4')]
    return render_template('results.html', segments=segments, output_dir=output_dir, full_video=app.config['FULL_VIDEO_OUTPUT'])

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/output_segments/<filename>')
def output_segment(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/full_video')
def full_video():
    return send_from_directory(app.config['OUTPUT_FOLDER'], app.config['FULL_VIDEO_OUTPUT'])

def process_video(video_path):
    model = YOLO("best_helmet_addy_yolov8s.pt")
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    region_points = [(0, 40), (565, 40), (565, 480), (0, 480)]
    full_video_writer = cv2.VideoWriter(app.config['FULL_VIDEO_OUTPUT'], cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    writing_segment = False
    segment_writer = None
    segment_counter = 0
    frame_no_hardhat = 0
    frame_limit = 120

    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break

        results = model.track(im0, persist=True, show=False, conf=0.5)
        boxes = results[0].boxes

        cv2.polylines(im0, [np.array(region_points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

        no_hardhat_detected = False

        if boxes is not None:
            for box in boxes:
                cls = box.cls.item()
                xyxy = box.xyxy[0]
                p1 = (int(xyxy[0]), int(xyxy[1]))
                p2 = (int(xyxy[2]), int(xyxy[3]))
                label = model.names[int(cls)]

                cv2.rectangle(im0, p1, p2, (255, 0, 0), 2)
                cv2.putText(im0, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                if label == 'NO-Hardhat':
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2
                    if cv2.pointPolygonTest(np.array(region_points, np.int32), (int(x_center), int(y_center)), False) >= 0:
                        no_hardhat_detected = True

        full_video_writer.write(im0)

        if no_hardhat_detected:
            frame_no_hardhat += 1
        else:
            frame_no_hardhat = 0

        if no_hardhat_detected and not writing_segment:
            segment_counter += 1
            segment_path = os.path.join(app.config['OUTPUT_FOLDER'], f"segment_{segment_counter}.mp4")
            segment_writer = cv2.VideoWriter(segment_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writing_segment = True
            print(f"Started new segment: {segment_path}")

        if writing_segment:
            if not no_hardhat_detected or frame_no_hardhat > frame_limit:
                segment_writer.release()
                writing_segment = False
                print(f"Stopped segment: {segment_path}")
            else:
                segment_writer.write(im0)

    cap.release()
    full_video_writer.release()
    if segment_writer is not None:
        segment_writer.release()
    cv2.destroyAllWindows()

    print("Processing complete. Check the output directories.")
    return app.config['OUTPUT_FOLDER']

if __name__ == '__main__':
    app.run(debug=True, port=5001)
