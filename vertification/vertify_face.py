import os
import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from scipy.spatial.distance import cosine

# ==== CẤU HÌNH ĐƯỜNG DẪN ====
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "recognition", "recognition", "face_classifier.h5")
FACES_JSON = os.path.join(BASE_DIR, "faces.json")
TEMP_IMG_PATH = os.path.join(BASE_DIR, "temp.jpg")

# ==== TẢI MÔ HÌNH ====
model = load_model(MODEL_PATH)

# ==== TRÍCH XUẤT ĐẶC TRƯNG (có cắt khuôn mặt) ====
def extract_embedding(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"[ERROR] Không thể đọc ảnh từ {img_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        raise ValueError("[ERROR] Không tìm thấy khuôn mặt trong ảnh.")

    x, y, w, h = faces[0]
    face = img[y:y + h, x:x + w]
    face = cv2.resize(face, (160, 160))

    img_array = image.img_to_array(face)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    embedding = model.predict(img_array)[0]
    return embedding

# ==== LƯU KHUÔN MẶT ====
def save_user_face(name, img_path, file=FACES_JSON):
    try:
        embedding = extract_embedding(img_path)
        try:
            with open(file, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append({"name": name, "embedding": embedding.tolist()})
        with open(file, 'w') as f:
            json.dump(data, f)
        print(f"[ĐÃ LƯU] {name} vào cơ sở dữ liệu.")
    except Exception as e:
        print(f"[ERROR] {str(e)}")

# ==== XÁC MINH KHUÔN MẶT ====
def verify_face(img_path, file=FACES_JSON, threshold=0.4):
    try:
        new_embedding = extract_embedding(img_path)
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return None

    try:
        with open(file, 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("[ERROR] Không tìm thấy cơ sở dữ liệu khuôn mặt.")
        return None

    for person in data:
        dist = cosine(new_embedding, person['embedding'])
        if dist < threshold:
            print(f"[MATCH] {person['name']} (khoảng cách: {dist:.2f})")
            return person['name']

    print("[KHÔNG KHỚP]")
    return None

# ==== TEST BẰNG CAMERA ====
def capture_and_verify(file=FACES_JSON):
    cap = cv2.VideoCapture(0)
    print("[INFO] Nhấn SPACE để chụp, ESC để thoát")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow('Xác minh khuôn mặt', frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == 32:
            cv2.imwrite(TEMP_IMG_PATH, frame)
            verify_face(TEMP_IMG_PATH, file=file)

    cap.release()
    cv2.destroyAllWindows()

# ==== TEST BẰNG ẢNH MÁY TÍNH ====
def test_from_file(img_path):
    if not os.path.exists(img_path):
        print(f"[ERROR] File không tồn tại: {img_path}")
        return
    result = verify_face(img_path)
    if result:
        print(f"[KẾT QUẢ] Người được xác minh: {result}")
    else:
        print("[KẾT QUẢ] Không tìm thấy kết quả khớp.")

# ==== MENU GIAO DIỆN DÒNG LỆNH ====
def main_menu():
    while True:
        print("\n=== HỆ THỐNG XÁC MINH KHUÔN MẶT ===")
        print("1. Thêm người dùng mới từ ảnh")
        print("2. Xác minh từ ảnh máy tính")
        print("3. Xác minh bằng webcam")
        print("0. Thoát")
        choice = input("Chọn: ")

        if choice == "1":
            name = input("Nhập tên người dùng: ")
            path = input("Nhập đường dẫn ảnh (ví dụ: D:\\images\\alice.jpg): ")
            if os.path.exists(path):
                save_user_face(name, path)
            else:
                print("[ERROR] File ảnh không tồn tại.")

        elif choice == "2":
            path = input("Nhập đường dẫn ảnh để xác minh: ")
            test_from_file(path)

        elif choice == "3":
            capture_and_verify()

        elif choice == "0":
            print("Thoát chương trình.")
            break

        else:
            print("Lựa chọn không hợp lệ. Vui lòng thử lại.")

# ==== CHẠY CHÍNH ====
if __name__ == "__main__":
    main_menu()
