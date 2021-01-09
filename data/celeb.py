import os
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms.transforms import Resize

from .utils import LinearNormalize


normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
linear_normalize_fn = LinearNormalize()
identity_normalize_fn = transforms.Lambda(lambda X: X)

def get_celeba(opt):
    norm_opt = opt.get('normalize', 'linear')
    if norm_opt == 'linear':
        norm = linear_normalize_fn
    elif norm_opt == 'identity':
        norm = identity_normalize_fn
    else:
        norm = normalize_fn

    training_transformer = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.CenterCrop(148),
                                transforms.Resize(opt.image_size),
                                transforms.ToTensor(),
                                norm
                        ])
    eval_transformer = training_transformer
    # eval_transformer = transforms.Compose([
    #                     transforms.Resize((64, 64)),
    #                     transforms.ToTensor(),
    #                     normalize,
    #                 ])

    data_root = os.path.join(opt.data_root, 'celeb')
    training_dataset = datasets.CelebA(root=data_root,
                                        split='train',
                                        transform=training_transformer,
                                        download=True)
    eval_dataset =  datasets.CelebA(root=data_root, 
                                     split='valid',
                                     download=True,
                                     transform=eval_transformer)

    return training_dataset, eval_dataset
