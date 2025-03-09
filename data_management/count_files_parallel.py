import os
from multiprocessing import Pool

# Function to count files in a directory
def count_files(directory):
    file_count = 0
    try:
        for foldername, _, filenames in os.walk(directory):
            file_count += len(filenames)
    except Exception as e:
        print(f"An error occurred while counting files in {directory}: {e}")
    return file_count


if __name__ == "__main__":
    directory = '/purestorage/datasets/laion_face_extracted/split_00001'

    # Get a list of all subdirectories
    subdirectories = [os.path.join(directory, d) for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    # If there are no subdirectories, count files in the top-level directory
    if not subdirectories:
        subdirectories = [directory]

    # Number of processors you want to use
    num_processors = 10  # Change this to the desired number of processors

    # Create a pool of worker processes
    with Pool(num_processors) as pool:
        # Count files in each subdirectory in parallel
        file_counts = pool.map(count_files, subdirectories)

    # Sum up the file counts from all subdirectories
    total_file_count = sum(file_counts)

    print(f"There are {total_file_count} files in the directory {directory} and its subdirectories.")