# MeRGAN Replay Alignment
python mergan.py --dataset mnist --RA --RA_factor 1e-3  --result_path mnist_RA_1e_3/
python mergan.py --dataset mnist --test  --result_path result/mnist_RA_1e_3/

python mergan.py --dataset svhn --RA --RA_factor 1e-2  --result_path svhn_RA_1e_2/
python mergan.py --dataset svhn --test  --result_path result/svhn_RA_1e_2/ 	

# MeRGAN Joint Training with Replay
python mergan.py --dataset mnist --JTR --result_path mnist_JTR/
python mergan.py --dataset mnist --test --result_path result/mnist_JTR/		

python mergan.py --dataset svhn --JTR --result_path svhn_JTR/
python mergan.py --dataset svhn --test --result_path result/svhn_JTR/

# Joint Training
python joint.py --dataset mnist --result_path mnist_joint/
python joint.py --dataset mnist --test --result_path result/mnist_joint/

python joint.py --dataset svhn --result_path svhn_joint/
python joint.py --dataset svhn --test --result_path result/svhn_joint/

# LSUN 
python mergan.py --dataset lsun --RA --RA_factor 1e-3 --result_path test_lsun_RA/
python mergan.py --dataset lsun --JTR --result_path test_lsun_RA/
