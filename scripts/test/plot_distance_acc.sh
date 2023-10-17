python save_features.py --dataset CUB --model Conv4NP --method protonet --train_aug --n_shot 1
CUDA_VISIBLE_DEVICES=2 python plot_distance_acc.py --dataset CUB --model Conv4NP --method protonet --train_aug --n_shot 1 --runs 1
CUDA_VISIBLE_DEVICES=2 python plot_distance_acc.py --dataset CUB --model Conv4NP --method protonet --train_aug --n_shot 1 --runs 2
CUDA_VISIBLE_DEVICES=2 python plot_distance_acc.py --dataset CUB --model Conv4NP --method protonet --train_aug --n_shot 1 --runs 3
CUDA_VISIBLE_DEVICES=2 python plot_distance_acc.py --dataset CUB --model Conv4NP --method protonet --train_aug --n_shot 1 --runs 4
CUDA_VISIBLE_DEVICES=2 python plot_distance_acc.py --dataset CUB --model Conv4NP --method protonet --train_aug --n_shot 1 --runs 5
