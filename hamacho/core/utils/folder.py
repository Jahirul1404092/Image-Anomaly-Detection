import os

def is_file(file_path):
    return os.path.isfile(file_path)


def list_folders(directory):
    folders = [ 
        folder for folder in os.listdir(directory) if not is_file(os.path.join(directory, folder))
    ]
    return folders


def file_exists(file_path):
    return os.path.exists(file_path)


def count_files(path, extensions=None):
    c = 0
    for filename in os.listdir(path):
        if not is_file(os.path.join(path, filename)): 
            continue
        _, file_ext = os.path.splitext(filename)
        if extensions is not None and file_ext.lower() not in extensions:
            continue
        c += 1

    return c


def compare_dir(dir1, dir2, extensions=None):
    items = {}
    for filename in os.listdir(dir1):

        if not is_file(os.path.join(dir1, filename)): 
            continue
        _, file_ext = os.path.splitext(filename)
        if extensions is not None and file_ext not in extensions:
            continue

        if not file_exists(os.path.join(dir2, filename)): 
            items[filename] = [
                dir1,  # found in dir1
                dir2,  # but not found in dir2
            ]

    return items