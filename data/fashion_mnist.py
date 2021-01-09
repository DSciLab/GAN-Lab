import os
from torchvision import datasets
from torchvision import transforms

from .utils import LinearNormalize


normalize_fn = transforms.Normalize(mean=[0.2860],
                                    std=[0.32045])
linear_normalize_fn = LinearNormalize()
identity_normalize_fn = transforms.Lambda(lambda X: X)


def get_fashion_mnist(opt):
    if opt.get('normalize', 'normalize') == 'linear':
        normalize = linear_normalize_fn
    else:
        normalize = normalize_fn

    training_transformer = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomCrop(opt.image_size, 4),
                            transforms.RandomRotation(15),
                            transforms.ToTensor(),
                            normalize,
                        ])
    eval_transformer = transforms.Compose([
                        transforms.ToTensor(),
                        normalize,
                    ])

    data_root = os.path.join(opt.data_root, 'fashion_mnist')
    training_dataset = datasets.FashionMNIST(root=data_root,
                                             train=True,
                                             transform=training_transformer,
                                             download=True)
    eval_dataset =  datasets.FashionMNIST(root=data_root, 
                                          train=False,
                                          download=True,
                                          transform=eval_transformer)

    return training_dataset, eval_dataset
