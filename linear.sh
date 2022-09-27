# nohup bash linear.sh > linear.out &


python linear.py \
--learning_rate 0.1 \
--test_epoch 1 \
--size 224 \
--device cuda:1 \
--test_dataset cifar100

python linear.py \
--learning_rate 0.1 \
--test_epoch 3 \
--size 224 \
--device cuda:1 \
--test_dataset cifar100

python linear.py \
--learning_rate 0.1 \
--test_epoch 5 \
--size 224 \
--device cuda:1 \
--test_dataset cifar100

python linear.py \
--learning_rate 0.1 \
--test_epoch 15 \
--size 224 \
--device cuda:1 \
--test_dataset cifar100

python linear.py \
--learning_rate 0.1 \
--test_epoch 60 \
--size 224 \
--device cuda:1 \
--test_dataset cifar100

python linear.py \
--learning_rate 0.1 \
--test_epoch 120 \
--size 224 \
--device cuda:1 \
--test_dataset cifar100

python linear.py \
--learning_rate 0.1 \
--test_epoch 210 \
--size 224 \
--device cuda:1 \
--test_dataset cifar100

python linear.py \
--learning_rate 0.1 \
--test_epoch 300 \
--size 224 \
--device cuda:1 \
--test_dataset cifar100