# CelebA
## Ours
python train_CelebA_our.py --experiment label --name debug --sample_interval 1
python train_CelebA_our.py --experiment label_mse --name debug --sample_interval 1
python train_CelebA_our.py --experiment label_mse_MI --name debug --sample_interval 1
# python train_CelebA_our.py --experiment label_mse_G1D --name debug --sample_interval 1
# python train_CelebA_our.py --experiment label_mse_G1pretrain --name debug --sample_interval 1
python train_CelebA_our.py --experiment label_mse_MI --name test
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name just_reconstruct_l1 --gc 1 --dc 1 --ga 0 --mi 0 --num_iter_MI 0
python train_CelebA_our.py --experiment label_mse_MI_Pscratch --name just_reconstruct_l1 --gc 1 --dc 1 --ga 0 --mi 0 --num_iter_MI 0


## test synthetic dataset
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name CelebA --data CelebA
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name CelebA_unbiased_929 --data CelebA_unbiased


## attGAN
python train_CelebA_our.py --experiment attGAN --name debug
python train_CelebA_our.py --experiment attGAN --name reproduce --shortcut_layers 1 --inject_layers 1 --lr 0.0002 --beta1 0.5 --beta2 0.999
python train_CelebA_our.py --experiment attGAN --name reproduce_tanh --shortcut_layers 1 --inject_layers 1 --lr 0.0002 --beta1 0.5 --beta2 0.999
python train_CelebA_our.py --experiment attGAN_MI --name baseline
python train_CelebA_our.py --experiment attGAN_MI --name 927_tanh
python train_CelebA_our.py --experiment attGAN_MI --name 927_sigmoid









## abalation study
python train_CelebA_our.py --experiment label_mse_MI --name mse_reconstruct
python train_CelebA_our.py --experiment label_mse_MI --name just_reconstruct --gc 0 --ga 0 --mi 0 --num_iter_MI 0
python train_CelebA_our.py --experiment label_mse_MI --name just_reconstruct_l1 --gc 0 --ga 0 --mi 0 --num_iter_MI 0
python train_CelebA_our.py --experiment label_mse_MI --name just_reconstruct_l1 --gc 1 --ga 0 --mi 0 --num_iter_MI 0
python train_CelebA_our.py --experiment label_mse_MI --name just_reconstruct_l1 --gc 0 --ga 1 --mi 1 --num_iter_MI 0











# attGAN
python train_CelebA_attGAN.py --experiment_name new_all 
python train_CelebA_attGAN.py --experiment_name new_male_and_hair --attrs Male Blond_Hair
python train_CelebA_attGAN.py --experiment_name new_male --attrs Male 
python train_CelebA_attGAN.py --experiment_name new_male_nogc --attrs Male --gc 0
python train_CelebA_attGAN.py --experiment_name new_male_nodc --attrs Male --dc 0
python train_CelebA_attGAN.py --experiment_name new_male_nodiscriminator --attrs Male --n_d 0
python train_CelebA_attGAN.py --experiment_name new_male_morediscriminator --attrs Male --n_d 100
python train_CelebA_attGAN.py --experiment_name new_male_lowgr --attrs Male --gr 10
python train_CelebA_attGAN.py --experiment_name new_male_1_02 --attrs Male --test_int 1 --thres_int 0.2
python train_CelebA_attGAN.py --experiment_name new_male_1_03 --attrs Male --test_int 1 --thres_int 0.3
python train_CelebA_attGAN.py --experiment_name new_male_2_03 --attrs Male --test_int 2 --thres_int 0.3
python train_CelebA_attGAN.py --experiment_name new_male_105_03 --attrs Male --test_int 1.5 --thres_int 0.3
python train_CelebA_attGAN.py --experiment_name new_male_105_02 --attrs Male --test_int 1.5 --thres_int 0.2
python train_CelebA_attGAN.py --experiment_name new_test --sample_interval 1

## IMDB
python train_IMDB_attGAN.py --experiment_name IMDB_test --sample_interval 1
python train_IMDB_attGAN.py --experiment_name IMDB_test_crop --sample_interval 1
python train_IMDB_attGAN.py --experiment_name IMDB_test_resize_crop --sample_interval 1

# attGAN + pretrain predictor/regressor
## CMNIST
python train_color.py --experiment_name color

# test
python test_multi.py --experiment_name all --test_atts Male --test_ints 0.5
python test_slide.py --experiment_name 224_shortcut1_inject1_none --test_att Male --test_int_min -1.0 --test_int_max 1.0 --n_slide 10 