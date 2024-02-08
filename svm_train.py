import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

def load_data(folder_path, num_images_per_class):
    images = []
    labels = []

    for class_label in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_label)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path)[:num_images_per_class]:
                img_path = os.path.join(class_path, filename)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, (100, 100))  # Adjust the size as needed
                    images.append(img)
                    labels.append(class_label)
                except Exception as e:
                    print(f"Error loading image: {img_path}, Error: {e}")

    return images, labels

def train_svm(train_folder, test_folder, num_images_per_class=50):
   
    # Loading subset of training data
    train_images, train_labels = load_data(train_folder, num_images_per_class)

    test_images, test_labels = load_data(test_folder, float('inf'))  # Load all test images

    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(train_images, train_labels)

    test_predictions = svm_model.predict(test_images)

    accuracy = accuracy_score(test_labels, test_predictions)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

    return svm_model

if __name__ == "__main__":
    train_dataset_folder = 'C:/Users/OneDrive/Desktop/ASL/data/asl_alphabet_train'
    test_dataset_folder = 'C:/Users/OneDrive/Desktop/ASL/data/asl_alphabet_test'
    trained_model = train_svm(train_dataset_folder, test_dataset_folder, num_images_per_class=50)
