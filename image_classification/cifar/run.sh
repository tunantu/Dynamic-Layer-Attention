python3 cifar.py -a dla_b --train-batch 128 --dataset cifar100 --depth 20 --block-name bottleneck --lr 0.1 --gpu-id 0,1  --epochs 180 --schedule 100 150 --drop-path 0.2   --gamma 0.1 --wd 1e-4 --checkpoint checkpoints/cifar100/resnet-110/dla-l-110



