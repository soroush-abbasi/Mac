# Mac


DeiT-small
```
python -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model mac_deit_small_patch16_224 --batch-size 256 --data-path /datasets/imagenet/ --lr 0.001 --mac-model --train-generator --generator-in-dim 32 --generator-out-dim 4096 --generator-lr 0.001 --weight-decay 0.00001 --output_dir /nfs_share3/code/soroush/checkpoint/mac_vit_s
```
