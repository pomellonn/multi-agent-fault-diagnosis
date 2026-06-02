import os
import shutil
import random

SOURCE_DIR = r"C:\Users\aleks\agents\universal_machinery_datasets\universal_machinery_datasets"
OUTPUT_DIR = r"C:\Users\aleks\agents\selected_dataset"

CLASS_FOLDERS = ["0_healthy", "1_bearing_faults"]

SAMPLES_PER_CLASS = 100
random.seed(42)

os.makedirs(OUTPUT_DIR, exist_ok=True)

for class_folder in CLASS_FOLDERS:
    source_class_path = os.path.join(SOURCE_DIR, class_folder)
    output_class_path = os.path.join(OUTPUT_DIR, class_folder)

    print(f"Checking: {source_class_path}")

    if not os.path.exists(source_class_path):
        raise FileNotFoundError(f"Can't find folder: {source_class_path}")

    os.makedirs(output_class_path, exist_ok=True)

    files = [
        f for f in os.listdir(source_class_path)
        if os.path.isfile(os.path.join(source_class_path, f))
    ]

    print(f"{class_folder}: found {len(files)} files")

    selected_files = random.sample(files, min(SAMPLES_PER_CLASS, len(files)))

    for file_name in selected_files:
        src = os.path.join(source_class_path, file_name)
        dst = os.path.join(output_class_path, file_name)
        shutil.copy2(src, dst)

    print(f"{class_folder}: copied {len(selected_files)} files")

print(f"Dataset saved to: {OUTPUT_DIR}")