# CelebA
## Ours
# python train_CelebA_our.py --experiment label --name debug --sample_interval 1
# python train_CelebA_our.py --experiment label_mse --name debug --sample_interval 1
# python train_CelebA_our.py --experiment label_mse_MI --name debug --sample_interval 1
# python train_CelebA_our.py --experiment label_mse_MI --name test
# python train_CelebA_our.py --experiment label_mse_MI_PDshare --name just_reconstruct_l1 --gc 1 --dc 1 --ga 0 --mi 0 --num_iter_MI 0
# python train_CelebA_our.py --experiment label_mse_MI_Pscratch --name just_reconstruct_l1 --gc 1 --dc 1 --ga 0 --mi 0 --num_iter_MI 0

python train_CelebA_our.py --experiment label_mse_MI_PDshare --name debug --gc 1 --dc 1 --ga 0 --mi 0 --num_iter_MI 0
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1012_finetuning

python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_finetuning --gr 150 --dim_per_attr 200 --thres_int 0.3 --batch_size 32
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name debug


python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_1_03 --thres_int 0.3
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_mse
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_0gr --gr 0
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_200gr --gr 200
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_0gc --gc 0
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_0dc --dc 0
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_0ga --ga 0
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_0mi --mi 0 
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_0num_iter_MI --num_iter_MI 0
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_1dim_per_attr --dim_per_attr 1
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_1000dim_per_attr --dim_per_attr 1000
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_0g1 --num_g1 0
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_0g2 --num_g2 0
python train_CelebA_our.py --experiment label_mse_MI_PDshare --name 1009_0dis --num_dis 0

python train_CelebA_our.py --experiment label_mse_MI_PDshare --name male_and_hair --attrs Male Blond_Hair


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



# IMDB
python train_IMDB_our_sex.py --experiment attGAN --name IMDB_debug --filter_train all
python train_IMDB_our_sex.py --experiment attGAN --name IMDB_train_eb1 --filter_train eb1
python train_IMDB_our_sex.py --experiment attGAN --name IMDB_train_eb2 --filter_train eb2
python train_IMDB_our_sex.py --experiment attGAN_MI --name IMDB_debug --filter_train all







## abalation study
python train_CelebA_our.py --experiment label_mse_MI --name mse_reconstruct
python train_CelebA_our.py --experiment label_mse_MI --name just_reconstruct --gc 0 --ga 0 --mi 0 --num_iter_MI 0
python train_CelebA_our.py --experiment label_mse_MI --name just_reconstruct_l1 --gc 0 --ga 0 --mi 0 --num_iter_MI 0
python train_CelebA_our.py --experiment label_mse_MI --name just_reconstruct_l1 --gc 1 --ga 0 --mi 0 --num_iter_MI 0
python train_CelebA_our.py --experiment label_mse_MI --name just_reconstruct_l1 --gc 0 --ga 1 --mi 1 --num_iter_MI 0











# attGAN
python train_CelebA_attGAN.py --experiment_name new_all 
python train_CelebA_attGAN.py --experiment_name new_male_and_glass --attrs Male Eyeglasses
python train_CelebA_attGAN.py --experiment_name new_male_and_hair --attrs Male Blond_Hair
python train_CelebA_attGAN.py --experiment_name new_male --attrs Male
python train_CelebA_attGAN.py --experiment_name new_male_00 --attrs Male --inject_layers 0 --shortcut_layers 0
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

## CMNIST
## attGAN + pretrain predictor/regressor
python train_CMNIST_attGAN.py --experiment attGAN --name CMNIST_1013
python train_CMNIST_attGAN.py --experiment attGAN_pretrain --name CMNIST_1013
python train_CMNIST_attGAN.py --experiment attGAN_pretrain --name CMNIST_1013_sigmoid
python train_CMNIST_attGAN.py --experiment attGAN_PDsplit --name CMNIST_1013
python train_CMNIST_attGAN.py --experiment attGAN_PDsplit --name CMNIST_1013 --gc 300 --dc 300

python train_CMNIST_our.py --experiment label_mse_MI_Pwarmup --name CMNIST_1013
python train_CMNIST_our.py --experiment label_mse_MI_Pwarmup --name CMNIST_1013_morewarmup10
python train_CMNIST_our.py --experiment label_mse_MI_Pwarmup --name CMNIST_1013_onetimewarmup26
python train_CMNIST_our.py --experiment label_mse_MI_Pwarmup --name CMNIST_1013_onetimewarmup26_2


## ours
python train_CMNIST_our.py --experiment label_mse_MI_PDshare --name CMNIST_-1 --biased_var -1 --gc 20
python train_CMNIST_our.py --experiment label_mse_MI_PDshare --name CMNIST_-1 --biased_var -1 --gc 200
python train_CMNIST_our.py --experiment label_mse_MI_PDshare --name CMNIST_-1 --biased_var -1 --dim_per_attr 10
python train_CMNIST_our.py --experiment label_mse_MI_PDshare --name CMNIST_-1 --biased_var -1 --dim_per_attr 50
python train_CMNIST_our.py --experiment label_mse_MI_PDshare --name CMNIST_-1_sigmoid
python train_CMNIST_our.py --experiment label_mse_MI_pretrain --name CMNIST_-1_sigmoid


















# test
python test_multi.py --experiment_name all --test_atts Male --test_ints 0.5
python test_slide.py --experiment_name 224_shortcut1_inject1_none --test_att Male --test_int_min -1.0 --test_int_max 1.0 --n_slide 10 