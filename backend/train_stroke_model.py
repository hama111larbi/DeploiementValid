import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, precision_score, recall_score

# 🔗 Dataset Paths
base_dir = r"C:\Users\LENOVO\Desktop\HEALTHCARE\DATA_nonstructurées\Brain_Stroke_CT-SCAN_image"
train_dir = os.path.join(base_dir, 'Train')
val_dir = os.path.join(base_dir, 'Validation')
test_dir = os.path.join(base_dir, 'Test')

# 📥 Load paths and labels
def load_image_paths_labels(folder_path):
    image_paths, labels = [], []
    for label in os.listdir(folder_path):
        class_dir = os.path.join(folder_path, label)
        if os.path.isdir(class_dir):
            for image_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, image_file)
                image_paths.append(img_path)
                labels.append(label)
    return image_paths, labels

train_paths, train_labels = load_image_paths_labels(train_dir)
val_paths, val_labels = load_image_paths_labels(val_dir)
test_paths, test_labels = load_image_paths_labels(test_dir)

# 📷 Preprocessing
IMG_SIZE = (224, 224)
def load_and_preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    return np.array(img) / 255.0

x_train = np.array([load_and_preprocess_image(p) for p in train_paths])
x_val   = np.array([load_and_preprocess_image(p) for p in val_paths])
x_test  = np.array([load_and_preprocess_image(p) for p in test_paths])

# 🔁 Labels → binary
lb = LabelBinarizer()
y_train = lb.fit_transform(train_labels)
y_val = lb.transform(val_labels)
y_test = lb.transform(test_labels)

# 🧮 Class Weights
train_labels_str = lb.inverse_transform(y_train)
weights = class_weight.compute_class_weight(class_weight='balanced',
                                            classes=np.unique(train_labels_str),
                                            y=train_labels_str)
class_weights = dict(enumerate(weights))
print("Class Weights:", class_weights)

# 📦 Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    fill_mode="nearest",
    horizontal_flip=False
)
train_generator = datagen.flow(x_train, y_train, batch_size=32)

# 🧠 VGG16 + Classifier
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in vgg_base.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(vgg_base.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model_vgg16 = Model(inputs=vgg_base.input, outputs=output)

# ⚙️ Compile & Train (Feature Extraction)
model_vgg16.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model_vgg16.fit(
    train_generator,
    validation_data=(x_val, y_val),
    epochs=10,
    callbacks=[early_stopping],
    class_weight=class_weights
)

# 🔓 Fine-tuning (Last 4 Layers)
for layer in vgg_base.layers[:-4]:
    layer.trainable = False
for layer in vgg_base.layers[-4:]:
    layer.trainable = True

model_vgg16.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

fine_tune_history = model_vgg16.fit(
    train_generator,
    validation_data=(x_val, y_val),
    epochs=30,
    callbacks=[early_stopping],
    class_weight=class_weights
)

# 💾 Save the model
model_vgg16.save('stroke_model.h5')
print("Model saved as 'stroke_model.h5'")

# 📊 Predictions
y_pred_probs = model_vgg16.predict(x_test)

# 🔻 Test various thresholds (Graph)
thresholds = np.arange(0.1, 0.9, 0.05)
recalls = []
precisions = []

for thresh in thresholds:
    y_pred_thresh = (y_pred_probs > thresh).astype(int)
    recalls.append(recall_score(y_test, y_pred_thresh))
    precisions.append(precision_score(y_test, y_pred_thresh))

plt.figure(figsize=(10, 5))
plt.plot(thresholds, recalls, label='Recall (Stroke)', marker='o')
plt.plot(thresholds, precisions, label='Precision (Stroke)', marker='x')
plt.axhline(y=0.92, color='red', linestyle='--', label='Target Recall = 0.92')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.title('Recall & Precision vs Threshold')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('threshold_analysis.png')
plt.close()

# 📌 Sélection d'un seuil (ex: 0.3)
threshold = 0.3
y_pred = (y_pred_probs > threshold).astype(int)

# 📝 Rapport final avec seuil ajusté
print(f"\nClassification Report @ Threshold = {threshold}")
print(classification_report(y_test, y_pred, target_names=["Normal", "Stroke"]))

# Save the threshold value
with open('model_threshold.txt', 'w') as f:
    f.write(str(threshold))
