import numpy as np
import os
from pathlib import Path


# This function counts directories, lists them alphabetically,
# shows how many files each has, and stores metadata.
def get_directory_metadata(path: str):
    """
    Retrieves metadata for all directories inside the specified path.

    Parameters:
    - path (str): The path to the directory to analyze.

    Returns:
    - list[dict]: List of metadata for each directory.
    - int: Total number of directories.
    - int: Total number of files across all directories.
    """
    metadata = []
    total_files = 0

    try:
        # List and sort directories alphabetically
        directories = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

        for d in directories:
            dir_path = os.path.join(path, d)
            file_list = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
            file_count = len(file_list)
            total_files += file_count

            # Store metadata for this directory
            metadata.append({
                "name": d,
                "file_count": file_count,
                "path": dir_path
            })

            # Print name with file count in parentheses
            print(f"{d} ({file_count})")

        return metadata, len(directories), total_files

    except Exception as e:
        print(f"Error retrieving metadata: {e}")
        return [], 0, 0


directory = Path(__file__).resolve().parents[1] / "tutors" / "tutor1" / "audio" / "spanish"

if __name__ == "__main__":
    metadata, dir_count, total_files = get_directory_metadata(str(directory))
    print(f"\nNumber of directories in '{directory}': {dir_count}")
    print(f"Total number of files across all directories: {total_files}")
