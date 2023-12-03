import inspect

class HyperParam:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Base:
    def save_hyperparameters(self, ignore=[]):
        frame=inspect.currentframe().f_back
        _, _, _, values = inspect.getargvalues(frame)
        self.hparams = {k: v for k, v in values.items() if k not in set(ignore+['self']) and not k.startswith('_')}
        for k,v in self.hparams.items():
            setattr(self, k, v)
