import json
import os
import re
import fnmatch
import matplotlib.pyplot as plt

# Function to extract the order number from a file name
def extract_order_number(filename):
    match = re.search(r"top_(\d+)", filename)
    return int(match.group(1)) if match else -1

# Pattern to match the specific files
# pattern = "alpaca-*top_*.json"
# directory = "../lm-evaluation-harness/results/"

pattern = "*.json"
directory = "../lm-evaluation-harness/all_15_top_k/"
# Dictionary to hold the average accuracies keyed by the order number
average_accuracies = {}

# Loop over the files in the directory
for filename in os.listdir(directory):
    if fnmatch.fnmatch(filename, pattern):  # Check if the file is a JSON file
        file_path = os.path.join(directory, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Extract accuracy values and compute their mean
        acc_values = [sub["acc"] for sub in data["results"].values() if "acc" in sub]
        average_acc = sum(acc_values) / len(acc_values) if acc_values else 0
        order_number = extract_order_number(filename)

        # Add to dictionary
        average_accuracies[order_number] = average_acc

# Sort the average accuracies by their order number
sorted_accuracies = dict(sorted(average_accuracies.items()))

# Plotting
order_numbers = sorted_accuracies.keys()
mean_accuracies = sorted_accuracies.values()

plt.figure(figsize=(10, 5))
plt.plot(order_numbers, mean_accuracies, marker='o')
plt.title('Mean Accuracy vs. Top K Experts')
plt.xlabel('Top K Experts')
plt.ylabel('Mean Accuracy')
plt.grid(True)
plt.savefig('mean_accuracies_plot_2.png')