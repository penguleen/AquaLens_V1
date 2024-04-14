import os

folder_path = os.path.join('.', 'ideal_class_images', 'stripes')  # Replace with the actual path to your image folder
new_prefix = "stripes_"

# List all files in the folder
files = os.listdir(folder_path)

# Filter out non-image files if needed
image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

# Rename each image file
for i, old_name in enumerate(image_files):
    extension = os.path.splitext(old_name)[1]
    new_name = f"{new_prefix}{i + 1:04d}{extension}"
    old_path = os.path.join(folder_path, old_name)
    new_path = os.path.join(folder_path, new_name)

    # Rename the file
    os.rename(old_path, new_path)

print("Image names have been successfully changed.")
