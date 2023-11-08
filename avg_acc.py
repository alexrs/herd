import json
import sys

# Check if the file path is provided as an argument
if len(sys.argv) < 2:
    print("Please provide the path to the JSON file.")
    sys.exit(1)

# Read the JSON data from the provided file path
file_path = sys.argv[1]
with open(file_path, 'r') as file:
    data = json.load(file)

# Extract acc values and compute their average
acc_values = [sub["acc"] for sub in data["results"].values() if "acc" in sub]
average_acc = sum(acc_values) / len(acc_values)
print(average_acc)
