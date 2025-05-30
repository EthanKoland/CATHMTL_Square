import os
import glob
import random


def generate_random_indices(length):
    indices = list(range(length))
    random.shuffle(indices)
    return indices


def split_list(input_list, indices, ratio=0.8):
    split_point = int(len(indices) * ratio)
    list1 = [input_list[i] for i in indices[:split_point]]
    list2 = [input_list[i] for i in indices[split_point:]]
    return list1, list2


data_root = 'data/XRayCathNew'
img_list = sorted(glob.glob(os.path.join(data_root, 'trainval', 'masks', '*.png')))

indices = generate_random_indices(len(img_list))

train_list, val_list = split_list(img_list, indices)

with open('data/XRayCathNew/trainval/imagesets/train.txt', mode='w', encoding='utf-8') as f:
    for idx, text_record in enumerate(sorted(train_list)):
        text_record = os.path.split(text_record)[-1][:-4]
        f.write(text_record)
        f.write('\n')

with open('data/XRayCathNew/trainval/imagesets/val.txt', mode='w', encoding='utf-8') as f:
    for idx, text_record in enumerate(sorted(val_list)):
        text_record = os.path.split(text_record)[-1][:-4]
        f.write(text_record)
        f.write('\n')

print()
