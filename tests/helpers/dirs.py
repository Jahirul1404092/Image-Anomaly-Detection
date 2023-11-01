import os


def create_dir(path):
    try:
        os.mkdir(path)
        return True
    except FileExistsError:
        return False


def create_dirs(path):
    os.makedirs(path, exist_ok=True)


def tree_printer(root):
    for root, dirs, files in os.walk(root):
        for d in dirs:
            print(os.path.join(root, d))    
        for f in files:
            print(os.path.join(root, f))
