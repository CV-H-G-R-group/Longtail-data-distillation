# python main_DM.py  --dataset CIFAR100-LT  --model ConvNet  --ipc 5  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init real  --lr_img 1  --num_exp 1  --num_eval 5 
# python main_DM.py  --dataset CIFAR100-head  --model ConvNet  --ipc 500  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init real  --lr_img 1  --num_exp 1  --num_eval 5 --partial_condense T

python main_DM.py  --dataset CIFAR100-head  --model ConvNet  --ipc 50  --aug_size 50 --dsa_strategy color_crop_cutout_flip_scale_rotate  --init real  --lr_img 1  --num_exp 1  --num_eval 5 --partial_condense T
# python main_DM.py  --dataset CIFAR10-head  --model ConvNet  --ipc 500  --aug_size 500 --dsa_strategy color_crop_cutout_flip_scale_rotate  --init real  --lr_img 1  --num_exp 1  --num_eval 5 --partial_condense T
