import os
import glob
import shutil

root_dir = 'data'
original_dir = os.path.join(root_dir, 'bak', 'segmasks')
new_dir = os.path.join(root_dir, 'XRayCath', 'trainval', 'masks')
img_list = sorted(glob.glob(os.path.join(original_dir, '*.jpg')))
for idx, img_file in enumerate(img_list):
    file_name = os.path.split(img_file)[-1][:-4]
    file_name = file_name.zfill(5)
    shutil.copy(img_file, os.path.join(new_dir, file_name + '.png'))
