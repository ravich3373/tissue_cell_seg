import numpy as np
from skimage.io import imsave
from tqdm import tqdm
import os
import cv2
from skimage.exposure import rescale_intensity
import gc
from patchify import patchify



def create_rgb_image(input_data, channel_colors):
    """Takes a stack of 1- or 2-channel data and converts it to an RGB image

    Args:
        input_data: 4D stack of images to be converted to RGB
        channel_colors: list specifying the color for each channel

    Returns:
        numpy.array: transformed version of input data into RGB version

    Raises:
        ValueError: if ``len(channel_colors)`` is not equal
            to number of channels
        ValueError: if invalid ``channel_colors`` provided
        ValueError: if input_data is not 4D, with 1 or 2 channels
    """

    if len(input_data.shape) != 4:
        raise ValueError('Input data must be 4D, '
                         f'but provided data has shape {input_data.shape}')

    if input_data.shape[3] > 2:
        raise ValueError('Input data must have 1 or 2 channels, '
                         f'but {input_data.shape[-1]} channels were provided')

    valid_channels = ['red', 'green', 'blue']
    channel_colors = [x.lower() for x in channel_colors]

    if not np.all(np.isin(channel_colors, valid_channels)):
        raise ValueError('Only red, green, or blue are valid channel colors')

    if len(channel_colors) != input_data.shape[-1]:
        raise ValueError('Must provide same number of channel_colors as channels in input_data')

    rgb_data = np.zeros(input_data.shape[:3] + (3,), dtype='float32')

    # rescale channels to aid plotting
    for img in range(input_data.shape[0]):
        for channel in range(input_data.shape[-1]):
            current_img = input_data[img, :, :, channel]
            non_zero_vals = current_img[np.nonzero(current_img)]

            # if there are non-zero pixels in current channel, we rescale
            if len(non_zero_vals) > 0:

                percentiles = np.percentile(non_zero_vals, [5, 95])
                rescaled_intensity = rescale_intensity(current_img,
                                                       in_range=(percentiles[0], percentiles[1]),
                                                       out_range='float32')

                # get rgb index of current channel
                color_idx = np.where(np.isin(valid_channels, channel_colors[channel]))
                rgb_data[img, :, :, color_idx] = rescaled_intensity

    # create a blank array for red channel
    return rgb_data


data_root = "/scratch/rc5124/datasets/lupus/amp/"


train_split = [("1150", "AMP_1150_Ch1Cy1-CD34,CD45,Fascin,b-catenin,Vimentin_input_image_merged.npy"),
               ("1154", "AMP_1154_Ch1Cy1-CD34,CD45,Fascin,b-catenin,Vimentin_input_image_merged.npy")]

val_split = [("1156", "AMP_1156_Ch1Cy1-CD34,CD45,Fascin,b-catenin,Vimentin_input_image_merged.npy")]
test_split = [("1158", "AMP_1158_Ch1Cy1-CD34,CD45,Fascin,b-catenin,Vimentin_input_image_merged.npy")]


splits = [train_split, val_split, test_split]
#train_data_dict = np.load(os.path.join(data_root, f"tissuenet_v1.1_train.npz"))
#val_data_dict = np.load(os.path.join(data_root, f"tissuenet_v1.1_val.npz"))
#test_data_dict = np.load(os.path.join(data_root, f"tissuenet_v1.1_test.npz"))

#data_dicts = {"train": {"X": train_data_dict["X"], "y": train_data_dict["y"]},
#                "val": {"X": val_data_dict["X"], "y": val_data_dict["y"]},
#                        "test": {"X": test_data_dict["X"], "y": test_data_dict["y"]}}


for split, split_fls in zip(["train", "val", "test"], [train_split, val_split, test_split]):
    for id, fl in split_fls:
        img = np.load(os.path.join(data_root, fl)).squeeze()
        split_dir = f"lupus_{split}"
        img_patches = patchify(img, (224,224,3), step=224)
        img_patches = img_patches.squeeze()
        rows, cols = img_patches.shape[:2]
        for r in tqdm(range(rows)):
            for col in range(cols):
                fl = f"lupus_{split}/{str(r).zfill(4)},{str(col).zfill(4)}.png"
                msk_fl = f"lupus_{split}/{str(r).zfill(4)},{str(col).zfill(4)}_msk.png"
                img_data = img_patches[r, col, ...].squeeze()
                cv2.imwrite(fl, img_data)
                mask = (img_data[:,:,2] > 0.75).astype(np.int8)
                cv2.imwrite(msk_fl, mask)
        del img
        gc.collect()

