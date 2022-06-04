OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 topformer_v2/train.py \
    --batch-size 16 --image-folder ../images/train/ \
    --df-path ../images/train.csv  \
    --checkpoint-interval 200 \
    --warmup-iters 1500 \
    --max-iters 20000 \
    --config topformer/conf_20k.py \
    --lr 0.00012 \
    --init-from ../TopFormer-B_512x512_4x8_160k-39.2.pth \

    
