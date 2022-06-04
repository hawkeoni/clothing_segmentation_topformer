import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

class Normalize_image(object):
    """Normalize given tensor into given mean and standard dev

    Args:
        mean (float): Desired mean to substract from tensors
        std (float): Desired std to divide from tensors
    """

    def __init__(self, mean, std):
        assert isinstance(mean, (float))
        if isinstance(mean, float):
            self.mean = mean

        if isinstance(std, float):
            self.std = std

        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)

        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)

        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

        else:
            assert "Please set proper channels! Normlization implemented only for 1, 3 and 18"


def transform_mask(orig_mask):
    background = [0, 1, 2, 3, 7, 9, 10, 12, 15, 16, 17, 18, 19, 20, 21, 28, 29, 32, 33, 34, 35, 36, 37,
             39, 41, 43, 44, 47, 52, 56, 57, 58]
    upper_body = [4, 5, 8, 11, 22, 24, 26, 38, 48, 49, 51, 54, 55]
    lower_body = [25, 27, 30, 31, 40, 42, 45, 53]
    whole_body = [6, 13, 14, 23, 46, 50]
    new_mask = np.zeros_like(orig_mask, dtype="uint8")
    background = np.isin(orig_mask, np.array(background))
    upper = np.isin(orig_mask, np.array(upper_body))
    lower = np.isin(orig_mask, np.array(lower_body))
    whole = np.isin(orig_mask, np.array(whole_body))
    new_mask = 1 * upper + 2 * lower + 3 * whole
    return new_mask


class ClothingCoParsing(Dataset):

    def __init__(self, dataset_dir: str, augs):
        self.dir = Path(dataset_dir)
        self.image_dir = self.dir / "jpeg_images/IMAGES"
        self.masks_dir = self.dir / "jpeg_masks/MASKS"
        self.image_paths = list(self.image_dir.glob("*"))
        self.augs = augs
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.masks_dir / str(image_path.name).replace("img", "seg")
        assert image_path.exists and mask_path.exists()
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (768, 768), interpolation=cv2.INTER_CUBIC)
        img.transpose((2, 0, 1))
        mask = cv2.imread(str(mask_path), 0)
        mask = cv2.resize(mask, (768, 768), interpolation=cv2.INTER_NEAREST)
        mask = transform_mask(mask)
        mask = torch.from_numpy(mask)
        if isinstance(self.augs, transforms.Compose):
            img = self.augs(img)
        elif isinstance(self.augs, alb.Compose):
            img = self.augs(image=img)["image"]
        else:
            raise ValueError(f"Unknown transforms type {type(self.augs)}")
        return img, mask



def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated: [start0] [length0] [start1] [length1]... in 1d array
    shape: (height,width) of array to return
    Returns numpy array according to the shape, 1 - mask, 0 - background
    """
    shape = (shape[1], shape[0])
    s = mask_rle.split()
    # gets starts & lengths 1d arrays
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    # gets ends 1d array
    ends = starts + lengths
    # creates blank mask image 1d array
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    # sets mark pixles
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    # reshape as a 2d mask image
    return img.reshape(shape).T  # Needed to align to RLE direction


class IMaterialistDataset(Dataset):

    def __init__(self, image_folder, csvname):
        self.image_folder = Path(image_folder) / "train"
        df = pd.read_csv(self.image_folder.parent / csvname)
        self.image_info = defaultdict(dict)
        df["CategoryId"] = df.ClassId.apply(lambda x: str(x).split("_")[0])
        temp_df = (
            df.groupby("ImageId")["EncodedPixels", "CategoryId"]
            .agg(lambda x: list(x))
            .reset_index()
        )
        size_df = df.groupby("ImageId")["Height", "Width"].mean().reset_index()
        temp_df = temp_df.merge(size_df, on="ImageId", how="left")
        for index, row in tqdm(temp_df.iterrows(), total=len(temp_df)):
            image_id = row["ImageId"]
            image_path = self.image_folder / image_id
            self.image_info[index]["image_path"] = image_path
            self.image_info[index]["labels"] = row["CategoryId"]
            self.image_info[index]["annotations"] = row["EncodedPixels"]
        
        self.augs = alb.Compose([
            alb.Resize(768, 768), 
            alb.HorizontalFlip(),
            alb.ShiftScaleRotate(p=0.1),
            alb.ColorJitter(p=0.1),
            alb.RandomBrightnessContrast(p=0.1),
            alb.RGBShift(p=0.1),
            alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])
    
    def __getitem__(self, idx):
        upperbody = [0, 1, 2, 3, 4, 5]
        lowerbody = [6, 7, 8]
        wholebody = [9, 10, 11, 12]
        markup = set(upperbody + lowerbody + wholebody)
        info = self.image_info[idx]
        image_path = info["image_path"]
        img = cv2.imread(str(image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        shape = [4] + list(img.shape[:2])
        mask = np.zeros(shape, dtype=np.uint8)
        for annotation, label in zip(info["annotations"], info["labels"]):
            label = int(label)
            if label not in markup:
                continue
            if label in upperbody:
                channel = 1
            elif label in lowerbody:
                channel = 2
            elif label in wholebody:
                channel = 3
            else:
                raise ValueError(f"Got unknown label {label}")
            label_mask = rle_decode(annotation, img.shape[:2])
            mask[channel] += label_mask
        mask = (mask > 0).astype("uint8")
        for i in range(4):
            mask[i] = mask[i] * i
        final_mask = mask.sum(axis=0) # H, W
        conflict = (final_mask > 3) 
        final_mask[conflict] = 1
        final_mask = final_mask.astype(np.uint8)
        aug = self.augs(image=img, mask=final_mask)
        return aug["image"], aug["mask"].long()


    def __len__(self):
        return len(self.image_info)
