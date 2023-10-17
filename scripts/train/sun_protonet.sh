CUDA_VISIBLE_DEVICES=2 python train.py --dataset SUN --model Conv4 --method protonet --train_aug --n_shot 1
CUDA_VISIBLE_DEVICES=2 python train.py --dataset SUN --model Conv4 --method protonet --train_aug --n_shot 5
