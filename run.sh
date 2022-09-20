# attGAN
python train.py --experiment_name all 
python train.py --experiment_name male_and_hair --attrs Male Blond_Hair
python train.py --experiment_name male --attrs Male 
python train.py --experiment_name male_nogc --attrs Male --gc 0
python train.py --experiment_name male_nodc --attrs Male --dc 0
python train.py --experiment_name male_nodiscriminator --attrs Male --n_d 0
python train.py --experiment_name male_1_02 --attrs Male --test_int 1 --thres_int 0.2
python train.py --experiment_name male_1_03 --attrs Male --test_int 1 --thres_int 0.3
python train.py --experiment_name male_2_03 --attrs Male --test_int 2 --thres_int 0.3
python train.py --experiment_name male_105_03 --attrs Male --test_int 1.5 --thres_int 0.3
python train.py --experiment_name male_105_02 --attrs Male --test_int 1.5 --thres_int 0.2
python train.py --experiment_name test --sample_interval 1

## IMDB
python train_attGAN_IMDB.py --experiment_name first_test

# attGAN + pretrain predictor/regressor
## CMNIST
python train_color.py --experiment_name color

## CelebA



# Ours
## CelebA
python train_CelebA.py --experiment label_mse --name debug --sample_interval 1
python train_CelebA.py --experiment label_mse --name 915_2
python train_CelebA.py --experiment label_mse --name 916_evening
python train_CelebA.py --experiment label_mse --name 919_evening_refactor

# test
python test_multi.py --experiment_name all --test_atts Male --test_ints 0.5
python test_slide.py --experiment_name 224_shortcut1_inject1_none --test_att Male --test_int_min -1.0 --test_int_max 1.0 --n_slide 10 