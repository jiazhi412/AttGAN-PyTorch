# origin
python train.py --experiment_name all 
python train.py --experiment_name male_and_hair --attrs Male Blond_Hair
python train.py --experiment_name male --attrs Male 
python train.py --experiment_name test --sample_interval 1

# CMNIST
python train_color.py --experiment_name color

# CelebA
python train_sex_label.py --experiment_name midautumn 

# test
python test_multi.py --experiment_name all --test_atts Male --test_ints 0.5
python test_slide.py --experiment_name 224_shortcut1_inject1_none --test_att Male --test_int_min -1.0 --test_int_max 1.0 --n_slide 10 