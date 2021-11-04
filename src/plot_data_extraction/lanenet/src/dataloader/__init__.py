from .tusimple import TuSimpleDataLoader, TuSimpleTestDataLoader
from .culane import CULaneDataLoader, CULaneTestDataLoader
from .bdd import BDDDataLoader, BDDTestDataLoader
from .dirloader import DirDataLoader

import torch.utils.data as data

datasets = {
	'tusimple': TuSimpleDataLoader,
	'tusimpletest': TuSimpleTestDataLoader,
	'culane': CULaneDataLoader,
	'bdd': BDDDataLoader,
	'culanetest': CULaneTestDataLoader,
	'dirloader': DirDataLoader,
}

def get_dataset(opt, **kwargs):
    if opt.loader_type == 'dataset':
        key = opt.dataset
    else:
        key = opt.loader_type

    return datasets[key](opt, **kwargs)

def get_data_loader(opt, **kwargs):

    dataset = get_dataset(opt, **kwargs)

    shuffle = kwargs['split'] == 'train'
    data_loader = data.DataLoader(dataset,
                                  batch_size=opt.batch_size,
                                  num_workers=opt.num_workers,
                                  shuffle=shuffle,
                                  pin_memory=True)
    return data_loader
