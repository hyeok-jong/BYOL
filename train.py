import os
import torch
from torchvision import models
import numpy as np
from tqdm import tqdm
import wandb
import time
import datetime
import torch.backends.cudnn as cudnn

from module import BYOL
from data import set_loader
from parser import BYOL_parser

from utils import AverageMeter
from utils import warmup_learning_rate, get_learning_rate, adjust_learning_rate
from utils import set_dir, save_logs
from utils import init_wandb
from utils import format_time


def set_model(args):
    if args.model == 'resnet18':
        model = models.resnet18(weights = None)
    elif args.model == 'resnet50':
        model = models.resnet50(weights = None)
    return model


def train(train_loader, model, optimizer, epoch, args):
    # 1 Epoch training.
    model.train()
    losses = AverageMeter()

    for idx, ((image1, image2), _) in enumerate(train_loader):
        image1 = image1.to(args.device)
        image2 = image2.to(args.device)


        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        loss = model(image1, image2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_moving_average()

        losses.update(loss.item(), image1.shape[0])
    
    res = {
        'training_loss' : losses.avg,
        'learning_rate' : get_learning_rate(optimizer)
    }
    return res

def main():
    set_dir()
    args = BYOL_parser()
    init_wandb(args)


    train_loader = set_loader(args)
    base_model = set_model(args)
    model = BYOL(
                base_model, 
                image_size = args.size, 
                hidden_layer = 'avgpool',
                projection_size = args.prediction_dim,
                projection_hidden_size = args.mid_dim,
                moving_average_decay = 0.99)

    model = model.to(args.device)
    cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)


    for epoch in tqdm(range(1, args.epochs + 1)):
        adjust_learning_rate(args, optimizer, epoch)

        start_time = time.time()
        res = train(train_loader, model, optimizer, epoch, args)
        loss = res['training_loss']
        lr = res['learning_rate']
        print(f'[epoch:{epoch}/{args.epochs}] [loss:{loss}] [lr:{np.round(lr,6)}] [Time:[{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]] [total time:{format_time(time.time() - start_time)}]')
        
        wandb.log(res, step = epoch)

        save_logs(args, epoch, base_model, optimizer, loss, lr)

    wandb.finish()

if __name__ == '__main__':
    main()