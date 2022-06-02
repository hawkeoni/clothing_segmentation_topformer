torchrun --nproc_per_node 4 topformer/train.py --batch-size 32 --image-folder ../images/train/ --df-path ../images/train.csv --lr 0.03 --checkpoint-interval 20
