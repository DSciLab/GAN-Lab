import os
import glob
import pathlib
import importlib
import inspect
from mlutils import Trainer
from mlutils import Log


def match_file(trainer):
    pwd = pathlib.Path(__file__).parent.absolute()
    sub_files = list(glob.glob(os.path.join(pwd, '*')))
    for file_path in sub_files:
        file = file_path.split('/')[-1]
        filename = ''.join(file.split('_')).split('.')[0]
        if filename.lower() == trainer.lower():
            return file.split('.')[0]

    raise RuntimeError(f'Can not match trainer ({trainer})')


def get_trainer(opt):
    trainer = opt.trainer
    file = match_file(trainer)
    Log.debug(f'get trainer file {file}')
    
    mod = importlib.import_module(f'trainer.{file}')
    for sub_mod_name in dir(mod):
        sub_mod = getattr(mod, sub_mod_name)
        if inspect.isclass(sub_mod) \
            and issubclass(sub_mod, Trainer) \
            and sub_mod.__name__ != Trainer.__name__:
            return sub_mod
    raise RuntimeError(f'Trianer not found ({file}).')
