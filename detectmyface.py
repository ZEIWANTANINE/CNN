import face_recognition
import cv2
import os

# Khởi tạo danh sách khuôn mặt đã biết
known_face_encodings = []
known_face_names = []

dataset_path = "dataset"

# Load dataset
print("🧠 Đang tải khuôn mặt từ thư mục dataset...")
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue

    for filename in os.listdir(person_folder):
        image_path = os.path.join(person_folder, filename)
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {image_path}: {e}")

print(f"✅ Đã load {len(known_face_encodings)} khuôn mặt.")

# Mở webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Không thể mở webcam.")
    exit()

print("🎥 Webcam đang chạy... Bấm ESC để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Không thể đọc từ webcam.")
        break

    # Thu nhỏ frame để xử lý nhanh hơn
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Chuyển BGR -> RGB

    # Tìm khuôn mặt và encoding (bọc trong try để tránh crash)
    try:
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations) if face_locations else []
    except Exception as e:
        print(f"❌ Lỗi khi lấy face encodings: {e}")
        face_locations = []
        face_encodings = []

    # Duyệt qua từng khuôn mặt tìm được
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        # Phóng to tọa độ về kích thước ban đầu
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Vẽ khung và tên
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Hiển thị frame
    cv2.imshow("Face Recognition", frame)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
