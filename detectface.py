import cv2
import mediapipe as mp
import os

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Tạo thư mục lưu ảnh
name = "alice"
os.makedirs(f"dataset/{name}", exist_ok=True)
count = 0

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Chuyển ảnh sang RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                # Cắt ảnh khuôn mặt
                face_img = frame[y:y+h, x:x+w]
                if face_img.size != 0:
                    filename = f"dataset/{name}/{count}.jpg"
                    cv2.imwrite(filename, face_img)
                    print(f"Saved: {filename}")
                    count += 1

                # Hiển thị khung
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27 or count >= 50:
            break

cap.release()
cv2.destroyAllWindows()