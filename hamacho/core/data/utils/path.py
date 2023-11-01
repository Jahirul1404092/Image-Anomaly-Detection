
from itertools import islice
from pathlib import Path

space = "    "
branch = "│   "
tee = "├── "
last = "└── "


# ref. https://stackoverflow.com/a/59109706/9215780
def get_directory_tree(
    dir_path: Path,
    level: int = -1,
    limit_to_directories: bool = False,
    category: str = None,
    length_limit: int = 1000,
):
    """Given a directory Path object print a visual tree structure"""
    dir_path = Path(dir_path)  # accept string coerceable to Path
    files = 0
    directories = 0
    temp = []

    def inner(dir_path: Path, prefix: str = "", level=-1):
        nonlocal files, directories

        if not level:
            return  # 0, stop iterating
        if limit_to_directories:
            contents = []
            for d in dir_path.iterdir():
                if not d.is_dir():
                    continue
                if category is not None and \
                    (d.name == category or d.parent.name == category):
                    contents.append(d)
        else:
            contents = list(dir_path.iterdir())

        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = branch if pointer == tee else space
                yield from inner(path, prefix=prefix + extension, level=level - 1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1

    view_true = f"{dir_path.name}"
    temp.append(dir_path.name)
    iterator = inner(dir_path, level=level)

    for line in islice(iterator, length_limit):
        if any(x in line for x in [".ipynb_checkpoints", "checkpoint"]):
            continue
        temp.append(line)

    if next(iterator, None):
        print(f"... length_limit, {length_limit}, reached, counted:")

    return """\n""".join(temp)
