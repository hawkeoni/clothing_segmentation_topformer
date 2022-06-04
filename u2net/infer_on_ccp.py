from time import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from net import U2NET
from utils import Normalize_image, load_checkpoint_mgpu, get_palette
from common_segmentation import calculate_iou, ClothingCoParsing



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main(args):
    palette = get_palette(4)
    print("Loading weights")
    model = U2NET(3, 4).eval().to(device)
    load_checkpoint_mgpu(model, args.load_path)
    print("Finish loading weights")
    img_transform = transforms.Compose([transforms.ToTensor(), Normalize_image(0.5, 0.5)])
    dataset = ClothingCoParsing(args.input_dir, augs=img_transform)
    ious = []
    from tqdm import tqdm
    for i in tqdm(range(len(dataset))):
        img, mask = dataset[i]
        image = img.unsqueeze(0).cuda()
        mask = mask.unsqueeze(0).cuda()
        original_size = image.shape[2:]
        with torch.no_grad():
            output = model(image)[0]
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

    # TODO: IOU
    
    args = parser.parse_args()
    main(args)