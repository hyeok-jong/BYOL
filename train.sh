nohup \
python train.py \
--batch_size 256 \
--size 224 \
--epochs 300 \
--learning_rate 0.003 \
--cosine \
--warm \
--model resnet18 \
--device cuda:1 \
--prediction_dim 256 \
--projection_dim 256 \
--mid_dim 4096 \
--dataset BAPPS \
> train.out &
