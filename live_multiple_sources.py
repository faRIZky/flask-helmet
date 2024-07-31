import cv2
from ultralytics import YOLO
from datetime import datetime

# Muat model YOLOv8
model_path = 'best_helmet_addy_yolov8n.pt'
model = YOLO(model_path)

# Buka kamera laptop dan webcam Logitech
cap_laptop = cv2.VideoCapture(0)  # Biasanya 0 adalah ID kamera laptop
cap_webcam = cv2.VideoCapture(1)  # Biasanya 1 adalah ID webcam eksternal

while True:
    ret1, frame1 = cap_laptop.read()
    ret2, frame2 = cap_webcam.read()

    if not ret1 or not ret2:
        print("Gagal membuka salah satu kamera")
        break

    # Deteksi objek di frame dari kamera laptop
    results1 = model(frame1)
    no_hardhats_laptop = 0
    for result in results1:
        for box in result.boxes:
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            if conf > 0.5:  # Hanya tampilkan deteksi dengan kepercayaan > 0.5
                if model.names[int(cls)] == "NO-Hardhat":
                    no_hardhats_laptop += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(frame1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame1, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if no_hardhats_laptop > 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - {no_hardhats_laptop} NO-Hardhats detected in Kamera Laptop")
    else:
        print("No NO-Hardhats detected in Kamera Laptop")

    # Deteksi objek di frame dari webcam Logitech
    results2 = model(frame2)
    no_hardhats_webcam = 0
    for result in results2:
        for box in result.boxes:
            conf = box.conf[0].cpu().numpy()
            cls = box.cls[0].cpu().numpy()
            if conf > 0.5:  # Hanya tampilkan deteksi dengan kepercayaan > 0.5
                if model.names[int(cls)] == "NO-Hardhat":
                    no_hardhats_webcam += 1
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(frame2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(frame2, f'{model.names[int(cls)]} {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if no_hardhats_webcam > 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} - {no_hardhats_webcam} NO-Hardhats detected in Webcam Logitech")
    else:
        print("No NO-Hardhats detected in Webcam Logitech")

    # Tampilkan frame yang telah diolah
    cv2.imshow('Kamera Laptop', frame1)
    cv2.imshow('Webcam Logitech', frame2)

    # Keluar dari loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan dan tutup jendela
cap_laptop.release()
cap_webcam.release()
cv2.destroyAllWindows()
