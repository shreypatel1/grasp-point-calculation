import os

def process_file(filepath):
    # Open the file and read all lines
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Modify each line according to the requirement
    modified_lines = []
    for line in lines:
        if line and line[0] == '0':
            modified_lines.append('2' + line[1:])
        elif line and line[0] == '2':
            modified_lines.append('0' + line[1:])
        else:
            modified_lines.append(line)

    # Write the modified lines back to the file
    with open(filepath, 'w') as file:
        file.writelines(modified_lines)

def process_directory(directory):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            process_file(filepath)
            print(f"Processed {filename}")

# Specify the directory you want to process
directory = "datasets/jenga-piece-detection.v1i.yolo11/valid/labels"  # Replace with your directory path
process_directory(directory)
