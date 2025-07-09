import os
import pandas as pd
import numpy as np
import cv2
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_DIR = 'images'
CSV_FILE = 'faces.csv'
IMG_SIZE = (160, 160)

print("[INFO] Reading CSV...")
df = pd.read_csv(CSV_FILE)

# Phải có cột 'label' trong faces.csv để train
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])

X = []
y = []

print("[INFO] Loading images...")
for _, row in df.iterrows():
    img_path = os.path.join(IMG_DIR, row['label'])  # đã sửa dòng này
    if not os.path.exists(img_path):
        print(f"[WARNING] Missing image: {img_path}")
        continue
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARNING] Cannot read image: {img_path}")
        continue
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    X.append(img)
    y.append(row['label_enc'])

X = np.array(X)
y = to_categorical(y)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

os.makedirs("recognition", exist_ok=True)
model.save("recognition/face_classifier.h5")
with open("recognition/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("[DONE] Recognition model saved.")
