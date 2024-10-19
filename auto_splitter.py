from ast import Continue, Return
import os
import shutil
from sklearn.model_selection import train_test_split

# Example: Path to the folder containing your images, organized by class
image_dir = 'C:/Users/Zak/Downloads/affectnet-hq'  # Modify this to your images' base directory
classes = os.listdir(image_dir)  # Assuming each class has a subfolder in 'image_dir'

# Create directories for the train-test split
base_dir = 'C:/Users/Zak/source/repos/CS731-2024/emotional-aware-chatbot-cs731-g02/Model_Training/inputs_and_outputs'
train_dir = os.path.join(base_dir, 'train_images_affect_autosplit')
test_dir = os.path.join(base_dir, 'test_images_affect_autosplit')

# Create base directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Create class subdirectories
for class_name in classes:
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

# Collect all image file paths and their corresponding labels (folder names)
image_paths = []
labels = []

for class_name in classes:

#   if class_name != 'disgust': # choose a specific class to load for debigging
#      continue

    class_folder = os.path.join(image_dir, class_name)
    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        image_paths.append(img_path)
        labels.append(class_name)

# Perform train-test split with stratification
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.2, stratify=labels, random_state=69)

# Function to copy images to the destination directory
def copy_images(image_paths, labels, destination_dir):
    for img_path, label in zip(image_paths, labels):
        # Define destination path for the image
        dest_folder = os.path.join(destination_dir, label)
        shutil.copy(img_path, dest_folder)

# Copy train and test images to their respective directories
copy_images(train_paths, train_labels, train_dir)
copy_images(test_paths, test_labels, test_dir)

print(f"Training and test image datasets saved in {base_dir}.")
