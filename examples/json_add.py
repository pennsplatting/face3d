import json
import sys

# Check if the JSON file and camera angle argument are provided
if len(sys.argv) != 3:
    print("Usage: python script.py your_file.json camera_angle_x")
    sys.exit(1)

# Get the JSON file path and camera angle from the command-line arguments
json_file_path = sys.argv[1]
camera_angle_x = float(sys.argv[2])  # Parse the camera angle as a float

# Load the existing JSON content from the file
try:
    with open(json_file_path, 'r') as json_file:
        existing_data = json.load(json_file)
except FileNotFoundError:
    print(f"File '{json_file_path}' not found.")
    sys.exit(1)

# Add or update the 'camera_angle_x' key in the JSON data
existing_data['camera_angle_x'] = camera_angle_x

# Write the updated JSON data back to the file
with open(json_file_path, 'w') as json_file:
    json.dump(existing_data, json_file, indent=4)

print(f"Updated '{json_file_path}' with 'camera_angle_x' value: {camera_angle_x}.")

