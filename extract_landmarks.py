import os
import json
import pandas as pd

ANNOTATION_DIR = r"C:\Users\SURFACE LAPTOP\Desktop\JABESS\ML\HaGRID\ann_train_val" 
MAX_SAMPLES_PER_CLASS = 300
OUTPUT_CSV = "keypoints_data.csv"

all_landmarks = []
all_labels = []

for filename in os.listdir(ANNOTATION_DIR):
    if not filename.endswith(".json"):
        continue

    gesture = filename.replace(".json", "")
    json_path = os.path.join(ANNOTATION_DIR, filename)

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Processing: {gesture}")
    count = 0

    for entry_id, ann in data.items():
        if count >= MAX_SAMPLES_PER_CLASS:
            break

        for landmarks, label in zip(ann["landmarks"], ann["labels"]):
            if len(landmarks) != 21:
                continue
            flat = [coord for point in landmarks for coord in point]  # flatten x, y
            all_landmarks.append(flat)
            all_labels.append(label)
            count += 1

print(f"Extracted {len(all_labels)} samples.")

# Save as CSV
df = pd.DataFrame(all_landmarks)
df["label"] = all_labels
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")

# To run: python extract_landmarks.py