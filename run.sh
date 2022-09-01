python train.py \
--img_size 128 \
--shortcut_layers 1 \
--inject_layers 1 \
--experiment_name male_only \
--gpu \
--attrs Male Blond_Hair

python train.py \
--img_size 128 \
--shortcut_layers 1 \
--inject_layers 1 \
--experiment_name male_only \
--gpu \
--attrs Male 

python train.py \
--img_size 128 \
--shortcut_layers 1 \
--inject_layers 1 \
--experiment_name blur_sex \
--gpu \
--attrs Male 

python test_multi.py \
--experiment_name 128_shortcut1_inject1_none \
--test_atts Male \
--test_ints 0.5 \
--gpu

python test_slide.py \
--experiment_name 128_shortcut1_inject1_none \
--test_att Male \
--test_int_min -1.0 \
--test_int_max 1.0 \
--n_slide 10 \
--gpu