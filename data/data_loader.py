from torchvision import transforms, datasets
import torch
from .utils import TwoCropTransform

def set_loader(args):
    # construct data loader
    if args.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif args.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

    
    normalize = transforms.Normalize(mean=mean, std=std)

    if args.distortion == 'supcon':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=args.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    elif args.distortion == 'rand':
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=args.size, scale=(0.2, 1.)),
            transforms.RandAugment(2, 9),
            transforms.ToTensor(),
            normalize,
        ])

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./datasets',
                                        transform=TwoCropTransform(train_transform),
                                        download=True)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./datasets',
                                        transform=TwoCropTransform(train_transform),
                                        download=True)

    train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader

