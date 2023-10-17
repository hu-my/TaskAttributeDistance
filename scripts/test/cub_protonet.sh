python save_features.py --dataset CUB --model Conv4NP --method protonet --train_aug --n_shot 1
CUDA_VISIBLE_DEVICES=3 python test.py --dataset CUB --model Conv4NP --method protonet --train_aug --n_shot 1
python save_features.py --dataset CUB --model Conv4NP --method protonet --train_aug --n_shot 5
CUDA_VISIBLE_DEVICES=3 python test.py --dataset CUB --model Conv4NP --method protonet --train_aug --n_shot 5