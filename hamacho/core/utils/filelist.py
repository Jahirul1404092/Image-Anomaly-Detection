import os
from collections import defaultdict
from torchvision.datasets.folder import IMG_EXTENSIONS
from hamacho.core.utils.folder import is_file, file_exists

# This function will take two list of image file path.
# It checks whether it's a file and exist.
def get_valid_filelist(good_lists, bad_lists, maks_lists, extensions=None):

    good_samples  = []  # return good list
    bad_samples   = []  # return bad list
    mask_samples  = []  # return mask list
    invalid_files = defaultdict(list)  # if anything invalid

    if extensions is None:
        extensions = IMG_EXTENSIONS

    if isinstance(extensions, str):
        extensions = (extensions,)
    
    extensions = tuple(ext.lower() for ext in extensions)
    
    # Checking for good items
    for gl in good_lists:
        if file_exists(gl) and is_file(gl):
            if gl.lower().endswith(extensions):
                good_samples.append(gl)
        else:
            invalid_files['good-file-list'].append(gl)
    
    # Checking for bad items
    for bl in bad_lists:
        if file_exists(bl) and is_file(bl):
            if bl.lower().endswith(extensions):
                bad_samples.append(bl)
        else:
            invalid_files['bad-file-list'].append(bl)
    
    # Checking for mask items [mask is optional]
    if maks_lists:
        for ml in maks_lists:
            if file_exists(ml) and is_file(ml):
                if ml.lower().endswith(extensions):
                    mask_samples.append(ml)
            else:
                invalid_files['mask-file-list'].append(ml)

    return good_samples, bad_samples, mask_samples, invalid_files


# This function seprates common and uncommon items \
# from two list which may contians multiple nested list each.
def match_unmatch(list1, list2):
    list1_val = sum(list1, [])
    list2_val = sum(list2, [])
    # cm_item = list(set(list1_val) & set(list2_val))
    ucm_item = list(set(list1_val) ^ set(list2_val))
    return ucm_item

# Bad image and its corresponding mask image should be paried.
# This fucntion checks and return the possible paired data (bad and mask) 
# And report unpaired data as well.
def get_valid_paired_filelist(bad_list, mask_list):
    # A dict container for {'/image_path/': [list_of_image_names_with_extention]}
    bl_paths = defaultdict(list)
    ml_paths = defaultdict(list)

    for i in bad_list:
        bl_paths[os.path.dirname(i)].append(os.path.basename(i))
    for j in mask_list:
        ml_paths[os.path.dirname(j)].append(os.path.basename(j))

    ucm_item = match_unmatch(bl_paths.values(), ml_paths.values())
    final_bad_list = []
    final_mask_list = []
    unpaired_bl_paths = []
    unpaired_ml_paths = []

    # Separate pair and unpair data for bad file list.
    for key, values in bl_paths.items():
        for value in values:
            full_path = os.path.join(key, value)
            if value not in ucm_item:
                final_bad_list.append(full_path)
            else:
                unpaired_bl_paths.append(full_path)

    # Separate pair and unpair data for mask file list.
    for key, values in ml_paths.items():
        for value in values:
            full_path = os.path.join(key, value)
            if value not in ucm_item:
                final_mask_list.append(full_path)
            else:
                unpaired_ml_paths.append(full_path)
    
    return (
        sorted(final_bad_list), 
        sorted(final_mask_list), 
        [unpaired_bl_paths, unpaired_ml_paths], 
    )