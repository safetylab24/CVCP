import json

# Function to load a JSON file
def load_json(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
        inspect_json_error(filename, e.pos)
        return None

# Function to inspect JSON file around the error position
def inspect_json_error(filename, error_position, context=50):
    with open(filename, 'r') as file:
        file.seek(max(error_position - context, 0))
        snippet = file.read(context * 2)
    print(f"Error position {error_position} in {filename}:\n{snippet}\n")

# Specify the path to your JSON file
filename = '/home/vrb230004/media/datasets/nuscenes2/v1.0-trainval/ego_pose.json'

# Load the JSON file and handle potential errors
sample_data = load_json(filename)

if sample_data is not None:
    # Count the number of items (objects) in the list
    total_items = len(sample_data)
    print(f"Total number of items in {filename}: {total_items}")
else:
    print(f"Failed to load JSON file {filename}. Check the error message for details.")
