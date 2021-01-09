import importlib


def get_gan(opt):
    try:
        mod = importlib.import_module(f'networks.gan.{opt.gan}')
    except ModuleNotFoundError:
        raise RuntimeError(
            f'Unrecognized gan {opt.gan}')

    return mod
