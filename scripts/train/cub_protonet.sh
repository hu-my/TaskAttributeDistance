CUDA_VISIBLE_DEVICES=3 python train.py --dataset CUB --model Conv4NP --method protonet --train_aug --n_shot 1
CUDA_VISIBLE_DEVICES=3 python train.py --dataset CUB --model Conv4NP --method protonet --train_aug --n_shot 5