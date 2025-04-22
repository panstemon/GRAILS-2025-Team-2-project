import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import plot_model
import tensorflow as tf

def load_images_from_folder(folder):
    images = []
    labels = []
    class_names = sorted(os.listdir(folder))
    for class_name in class_names:
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            file_path = os.path.join(class_path, filename)
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.resize(img, (32, 32))
                img = img.astype('float32') / 255
                images.append(img)
                labels.append(class_names.index(class_name))
            else:
                print(f"Could not read file: {file_path}")
    return (
        np.array(images), 
        to_categorical(np.array(labels), len(class_names)), 
        labels, 
        class_names
    )


dataset_folder = '../datasets/persons/'


x_data, y_data, y_labels, class_names = load_images_from_folder(dataset_folder)
num_classes = len(class_names)


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

class_weights = dict(enumerate(compute_class_weight(class_weight='balanced', classes=np.unique(y_labels), y=y_labels)))


model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=15,
    verbose=1,
    validation_data=(x_test, y_test),
    class_weight=class_weights
)


model.save('../models/my_model.keras')


def update_class_names_file(class_names, file_path='../datasets/persons/class_names.txt'):
    with open(file_path, 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
update_class_names_file(class_names)




