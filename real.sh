CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 topformer/train.py --batch-size 32 --image-folder ../images/train/ --df-path ../images/train.csv 
