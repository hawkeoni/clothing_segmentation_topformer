torchrun --nproc_per_node 4 topformer/train.py --batch-size 32 --image-folder ../images/train/ --df-path ../images/train.csv --lr 0.003 --checkpoint-interval 20 --max-iters 400 --warmup-iters 50
