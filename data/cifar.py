import os
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms.transforms import Resize

from .utils import LinearNormalize


normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
linear_normalize_fn = LinearNormalize()
identity_normalize_fn = transforms.Lambda(lambda X: X)


def get_cifar10(opt):
    norm_opt = opt.get('normalize', 'linear')
    if norm_opt == 'linear':
        norm = linear_normalize_fn
    elif norm_opt == 'identity':
        norm = identity_normalize_fn
    else:
        norm = normalize_fn

    training_transformer = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.Resize(opt.image_size),
                            transforms.RandomCrop(opt.image_size, 4),
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            norm,
                        ])
    eval_transformer = transforms.Compose([
                        transforms.Resize(opt.image_size),
                        transforms.ToTensor(),
                        norm,
                    ])

    data_root = os.path.join(opt.data_root, 'cifar10')
    training_dataset = datasets.CIFAR10(root=data_root,
                                        train=True,
                                        transform=training_transformer,
                                        download=True)
    eval_dataset =  datasets.CIFAR10(root=data_root, 
                                     train=False,
                                     download=True,
                                     transform=eval_transformer)

    return training_dataset, eval_dataset



def get_cifar100(opt):
    norm_opt = opt.get('normalize', 'linear')
    if norm_opt == 'linear':
        norm = linear_normalize_fn
    elif norm_opt == 'identity':
        norm = identity_normalize_fn
    else:
        norm = normalize_fn

    training_transformer = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(opt.image_size, 4),
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            norm,
                        ])
    eval_transformer = transforms.Compose([
                        transforms.ToTensor(),
                        norm,
                    ])

    data_root = os.path.join(opt.data_root, 'cifar100')
    training_dataset = datasets.CIFAR100(root=data_root,
                                         train=True,
                                         transform=training_transformer,
                                         download=True)
    eval_dataset =  datasets.CIFAR100(root=data_root, 
                                      train=False,
                                      download=True,
                                      transform=eval_transformer)

    return training_dataset, eval_dataset
