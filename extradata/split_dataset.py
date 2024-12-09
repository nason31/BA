import os
import random

# Input and output paths
input_file = "R52/r52-raw.txt"  # Path to the single R52 dataset file
output_dir = "R52"  # Path to the directory where train.txt and test.txt will be saved
os.makedirs(output_dir, exist_ok=True)
print("test")

train_file = os.path.join(output_dir, "train.txt")
test_file = os.path.join(output_dir, "test.txt")

# Split ratio (e.g., 90% train, 10% test)
split_ratio = 0.9

# Read the dataset
with open(input_file, "r") as f:
    lines = f.readlines()

# Shuffle and split the data
random.shuffle(lines)
split_index = int(len(lines) * split_ratio)
train_lines = lines[:split_index]
test_lines = lines[split_index:]

# Save the split datasets
with open(train_file, "w") as f:
    f.writelines(train_lines)

with open(test_file, "w") as f:
    f.writelines(test_lines)

print(f"Dataset split complete. Train: {len(train_lines)} lines, Test: {len(test_lines)} lines.")
print(f"Train file: {train_file}")
print(f"Test file: {test_file}")
