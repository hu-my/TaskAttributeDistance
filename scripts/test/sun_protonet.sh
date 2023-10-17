python save_features.py --dataset SUN --model Conv4 --method protonet --train_aug --n_shot 1
CUDA_VISIBLE_DEVICES=2 python test.py --dataset SUN --model Conv4 --method protonet --train_aug --n_shot 1
python save_features.py --dataset SUN --model Conv4 --method protonet --train_aug --n_shot 5
CUDA_VISIBLE_DEVICES=2 python test.py --dataset SUN --model Conv4 --method protonet --train_aug --n_shot 5