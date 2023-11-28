import numpy as np
from skimage.io import imsave
from tqdm import tqdm
import os
import cv2


data_root = "/scratch/rc5124/datasets/tissuenet"
train_data_dict = np.load(os.path.join(data_root, f"tissuenet_v1.1_train.npz"))
val_data_dict = np.load(os.path.join(data_root, f"tissuenet_v1.1_val.npz"))
test_data_dict = np.load(os.path.join(data_root, f"tissuenet_v1.1_test.npz"))

data_dicts = {"train": {"X": train_data_dict["X"], "y": train_data_dict["y"]},
                "val": {"X": val_data_dict["X"], "y": val_data_dict["y"]},
                        "test": {"X": test_data_dict["X"], "y": test_data_dict["y"]}}


for split in ["train", "val", "test"]:
    split_dir = f"{split}"
    for i in tqdm(range(len(data_dicts[split]["X"]))):
        fl = f"{split}/{str(i).zfill(5)}.png"
        msk_fl = f"{split}/{str(i).zfill(5)}_msk.png"
        img_data = data_dicts[split]["X"][i,:,:,0]
        cv2.imwrite(fl, img_data)
        msk_data = data_dicts[split]["y"][i,:,:,0]
        cv2.imwrite(msk_fl, msk_data)

