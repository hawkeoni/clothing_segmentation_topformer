from time import time
from argparse import ArgumentParser
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import albumentations as alb
from albumentations.pytorch.transforms import ToTensorV2

from saveload import load_model
from data import Normalize_image
from common_segmentation import calculate_iou, get_palette
from common_segmentation import ClothingCoParsing



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main(args):
    print("Loading weights")
    model = load_model(args.load_path)
    print("Finish loading weights")
    dataset = ClothingCoParsing(args.input_dir,
    augs = alb.Compose([
            alb.Resize(768, 768), 
            alb.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
            ])
    )
    ious = []
    from tqdm import tqdm
    for i in tqdm(range(len(dataset))):
        img, mask = dataset[i]
        image = img.unsqueeze(0).cuda()
        mask = mask.unsqueeze(0).cuda()
        original_size = image.shape[2:]
        with torch.no_grad():
            output = model(image)
            # output - batch, classes, h, w
            output = F.upsample(output, original_size, mode="bilinear") 
        ious.append(calculate_iou(output, mask))
    ious = np.array(ious)
    print(np.mean(ious, axis=0))



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True,
        help="Directory to read images from")
    parser.add_argument("--output-dir", type=Path, required=False, help="")
    parser.add_argument("--load-path", type=Path, required=True,
        help="Model weights to load.")
    parser.add_argument("--iou", action="store_true", help="Calculate mIOU.")

    args = parser.parse_args()
    main(args)