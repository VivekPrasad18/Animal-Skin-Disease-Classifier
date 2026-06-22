import os
import shutil
import random

# ============================
# CONFIG
# ============================
TRAIN_PATH = "../dataset/train"
VALID_PATH = "../dataset/valid"
TEST_PATH = "../dataset/test"

SAMPLES_PER_CLASS = 10  # safe number

# ============================
# FUNCTION
# ============================
def fix_dataset(split_path):
    train_classes = os.listdir(TRAIN_PATH)

    for cls in train_classes:
        train_cls_path = os.path.join(TRAIN_PATH, cls)
        split_cls_path = os.path.join(split_path, cls)

        # Skip non-directories
        if not os.path.isdir(train_cls_path):
            continue

        # Create folder if missing
        if not os.path.exists(split_cls_path):
            os.makedirs(split_cls_path)
            print(f"Created folder: {split_cls_path}")

        # Get images
        images = os.listdir(train_cls_path)

        # Shuffle and pick limited samples
        random.shuffle(images)
        selected_images = images[:SAMPLES_PER_CLASS]

        for img in selected_images:
            src = os.path.join(train_cls_path, img)
            dst = os.path.join(split_cls_path, img)

            # Avoid overwriting
            if not os.path.exists(dst):
                shutil.copy(src, dst)

    print(f"\nDone fixing: {split_path}\n")


# ============================
# RUN
# ============================
if __name__ == "__main__":
    print("Fixing VALID dataset...")
    fix_dataset(VALID_PATH)

    print("Fixing TEST dataset...")
    fix_dataset(TEST_PATH)

    print("Dataset is now consistent 🚀")