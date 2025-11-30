import tensorflow as tf   
from tensorflow.keras import datasets
import numpy as np
from PIL import Image
import os
import random


class_names = ['cat', 'deer', 'airplane', 'automobile', 'bird', 'ship']
cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

desired_labels = [cifar10_classes.index(name) for name in class_names]
print("Desired labels:", desired_labels)

(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_labels = train_labels.flatten()
test_labels = test_labels.flatten()


selected_images = []
selected_labels = []

def get_images(test_dir = "./data", count=50, cls_name=[]):
    os.makedirs(test_dir, exist_ok=True)
    for cls in cls_name:
        dir = os.path.join(test_dir, cls)
        os.makedirs(dir, exist_ok=True)

    for label in desired_labels:
        indices = np.where(train_labels == label)[0]
        print("len", len(indices))
        rnd = random.randint(0, 3000)
        selected_indices = indices[rnd:rnd+count]
        cls_name = cifar10_classes[label]
        dir_path = os.path.join(test_dir, cls_name)

        for i, idx in enumerate(selected_indices):
            img = train_images[idx]
            img_pil = Image.fromarray(img)
            img_pil.save(os.path.join(dir_path, f"{cls_name}_{i}.png"))


if __name__ == "__main__":
    test_dir = "./data/test"
    train_dir = "./data/train"
    get_images(test_dir, 50, class_names)
    get_images(train_dir, 20, class_names)