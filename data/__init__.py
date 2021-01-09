from .mnist import get_mnist
from .cifar import get_cifar10, get_cifar100
from .fashion_mnist import get_fashion_mnist
from .celeb import get_celeba


def get_data(opt):
    if opt.dataset == 'MNIST':
        return get_mnist(opt)
    if opt.dataset == 'CELEBA':
        return get_celeba(opt)
    elif opt.dataset == 'CIFAR10':
        return get_cifar10(opt)
    elif opt.dataset == 'CIFAR100':
        return get_cifar100(opt)
    elif opt.dataset == 'FASHIONMNIST':
        return get_fashion_mnist(opt)
    else:
        raise RuntimeError(
            f'Unrecognized dataset [{opt.dataset}]')
