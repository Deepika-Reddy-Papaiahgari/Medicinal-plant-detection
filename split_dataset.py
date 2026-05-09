import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Original dataset path
SOURCE_DIR = "Datasets\Medicinal plant dataset"

# New dataset folders
BASE_DIR = "dataset"

# Split ratios
TRAIN_SIZE = 0.7
VAL_SIZE = 0.15
TEST_SIZE = 0.15
# Create folders
for folder in ['train', 'validation', 'test']:
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)

# Process each class
for class_name in os.listdir(SOURCE_DIR):

    class_path = os.path.join(SOURCE_DIR, class_name)

    if os.path.isdir(class_path):

        images = [img for img in os.listdir(class_path)
          if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(class_name, len(images))

        random.shuffle(images)

        # Split train and temp
        train_images, temp_images = train_test_split(
            images,
            test_size=(1 - TRAIN_SIZE),
            random_state=42
        )

        # Split validation and test
        val_images, test_images = train_test_split(
            temp_images,
            test_size=0.5,
            random_state=42
        )

        # Create class folders
        for folder in ['train', 'validation', 'test']:
            os.makedirs(os.path.join(BASE_DIR, folder, class_name), exist_ok=True)

        # Copy train images
        for image in train_images:
            src = os.path.join(class_path, image)
            dst = os.path.join(BASE_DIR, 'train', class_name, image)
            shutil.copyfile(src, dst)

        # Copy validation images
        for image in val_images:
            src = os.path.join(class_path, image)
            dst = os.path.join(BASE_DIR, 'validation', class_name, image)
            shutil.copyfile(src, dst)

        # Copy test images
        for image in test_images:
            src = os.path.join(class_path, image)
            dst = os.path.join(BASE_DIR, 'test', class_name, image)
            shutil.copyfile(src, dst)

print("Dataset split completed successfully!")