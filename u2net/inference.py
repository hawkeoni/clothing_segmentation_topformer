from time import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from net import U2NET
from utils import Normalize_image, load_checkpoint_mgpu, get_palette
from common_segmentation import calculate_multiclass_iou



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def main(args):
    palette = get_palette(4)
    img_transform = transforms.Compose([transforms.ToTensor(), Normalize_image(0.5, 0.5)])
    print("Loading weights")
    model = U2NET(3, 4).eval().to(device)
    load_checkpoint_mgpu(model, args.load_path)
    print("Finish loading weights")

    ious = []

    tottime = 0
    for image_num, image_path in tqdm(enumerate(Path(args.input_dir).glob("*"), start=1)):
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        image = image.resize((768, 768), Image.BICUBIC)
        x = img_transform(image).unsqueeze(0).to(device).half()
        with torch.no_grad():
            start = time()
            output = model(x)[0]
            end = time()
            tottime += end - start
            # output - batch, classes, h, w
            output = output.squeeze(0).argmax(0)

        # if args.iou:
        # calculate_multiclass_iou(true_labels, output, ["background", "upper_body", "lower_body", "whole_body"])
        if args.output_dir:
            output_arr = output.cpu().numpy()
            output_img = Image.fromarray(output_arr.astype('uint8'), mode='L')
            output_img = output_img.resize(original_size, Image.BICUBIC)
            output_img.putpalette(palette)
            output_img.save(args.output_dir / image_path.name)
        

        if image_num == 100:
            break


    print(f"It took {tottime} seconds to process {image_num} images. (Average {tottime / image_num} seconds per image)")


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