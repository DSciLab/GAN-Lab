import os
from torchvision import datasets
from torchvision import transforms

from .utils import LinearNormalize


normalize_fn = transforms.Normalize(mean=[0.1307],
                                 std=[0.30150])
linear_normalize_fn = LinearNormalize()
identity_normalize_fn = transforms.Lambda(lambda X: X)


def get_mnist(opt):
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

    data_root = os.path.join(opt.data_root, 'mnist')
    training_dataset = datasets.MNIST(root=data_root,
                                      train=True,
                                      transform=training_transformer,
                                      download=True)
    eval_dataset =  datasets.MNIST(root=data_root,
                                   train=False,
                                   download=True,
                                   transform=eval_transformer)

    return training_dataset, eval_dataset
