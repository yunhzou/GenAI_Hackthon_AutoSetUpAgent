import os
import fnmatch

def load_ignore_patterns(ignore_file=".ignore"):
    """
    Load ignore patterns from the specified .ignore file.
    """
    ignore_patterns = []
    if os.path.exists(ignore_file):
        with open(ignore_file, "r") as file:
            for line in file:
                line = line.strip()
                if line and not line.startswith("#"):  # Ignore empty lines and comments
                    ignore_patterns.append(line)
    return ignore_patterns

def is_ignored(path, ignore_patterns):
    """
    Check if the given path matches any of the ignore patterns.
    """
    for pattern in ignore_patterns:
        # Match directories with trailing `/`
        if pattern.endswith("/") and os.path.isdir(path) and fnmatch.fnmatch(path, f"*{pattern.rstrip('/')}*"):
            return True
        # Match files or general patterns
        elif fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False


def generate_tree_structure(base_path, ignore_patterns, prefix="", root_path=None):
    """
    Recursively generate a tree structure as a string, excluding ignored files/folders.
    For files:
      - If the file is in the root (base_path == root_path), add a hint to organize it.
      - Otherwise, specify which folder the file belongs to.
    """
    # Track the original root path on the first call
    if root_path is None:
        root_path = base_path

    if not os.path.exists(base_path):
        return "Currently the directory is empty"

    tree = []
    items = sorted(os.listdir(base_path))

    # Filter out ignored files and folders
    items = [
        item for item in items
        if not is_ignored(os.path.join(base_path, item), ignore_patterns)
    ]

    for index, item in enumerate(items):
        full_path = os.path.join(base_path, item)
        connector = "└── " if index == len(items) - 1 else "├── "
        folder_name = os.path.basename(base_path)

        if os.path.isdir(full_path):
            # Show directory in the tree
            tree.append(f"{prefix}{connector}{item}")
            # Recurse into this directory
            new_prefix = prefix + ("    " if index == len(items) - 1 else "│   ")
            tree.append(generate_tree_structure(full_path, ignore_patterns, new_prefix, root_path))
        else:
            # It's a file
            if base_path == root_path:
                # If it's directly under the root
                tree.append(f"{prefix}{connector}{item} (located in root)")
            else:
                # If it's inside a subfolder
                tree.append(f"{prefix}{connector}{item} (located in '{folder_name}' folder)")

    if not tree:
        return "Currently the directory is empty"

    return "\n".join(tree)
