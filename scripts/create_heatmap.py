import cv2
import numpy as np
import pandas as pd

IMAGE_SIZE = 512
SCALE_VECTOR = [IMAGE_SIZE, IMAGE_SIZE, 2 * np.pi, IMAGE_SIZE, IMAGE_SIZE]


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x**2 + y**2) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


# Read Blob label csv file
bdf = pd.read_csv('data/XRayCath/trainval/csv/blob.csv', delimiter=',')
bdf_length = int(bdf.iloc[-1]['imageId'][:-4]) + 1

for counter in range(0, bdf_length):
    print(counter)
    blackImg = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype="uint8")
    xy = np.zeros((2, ), dtype="float")
    blackImg = blackImg / 255.0

    # Create gaussian map for blobs
    imgname = (f"{(counter):04d}.jpg")
    blob_df = bdf.loc[bdf['imageId'] == imgname]
    for i in range(len(blob_df)):
        tx = int(blob_df["Center X"].iloc[i] * IMAGE_SIZE)
        ty = int(blob_df["Center Y"].iloc[i] * IMAGE_SIZE)
        strength = blob_df["Strength"].iloc[i]
        xy[0] = tx
        xy[1] = ty
        # print(xy, strength)
        if strength > 0.7:
            draw_umich_gaussian(blackImg[:, :, 0], xy, 12)
        else:
            draw_umich_gaussian(blackImg[:, :, 0], xy, 8)
    outname = f"data/XRayCath/trainval/heatmaps/{counter:05d}.png"
    blackImg = blackImg * 255
    cv2.imwrite(outname, blackImg)
