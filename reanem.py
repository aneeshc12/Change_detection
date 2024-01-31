import os
import shutil

def move_and_rename_images(root_folder):

    for root, dirs, files in os.walk(root_folder):
        for dir_name in dirs:
            counter = 0
            source_folder = os.path.join(root, dir_name)

            # Get a list of image files in the subfolder
            image_files = [file for file in os.listdir(source_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

            # Move and rename the image files to the root directory
            for image_file in image_files:
                counter += 1
                source_path = os.path.join(source_folder, image_file)
                new_name = f"{counter}.png"
                destination_path = os.path.join(source_folder, new_name)

                # Use shutil.move to perform the move operation
                shutil.move(source_path, destination_path)
                print(f"Moved and Renamed: {source_path} -> {destination_path}")

if __name__ == "__main__":
    root_folder = "/home2/aneesh.chavan/p/"  # Replace with the actual path to your root folder
    for i in ["armchairs",  "beds",  "chairs",  "coffee_tables",  "dining_tables",  "sofas",  "tv_stands"]:
        print(i)
        move_and_rename_images(root_folder + i)