# File for helper classes and methods

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


# *** Exceptions ***

class UnknownObjectTypeException(BaseException):
    pass


class UnknownPropertyException(BaseException):
    pass


# *** Definitions ***

IMG_SIZE = 128
NUM_REGIONS = 8

USE_GPU = True
DTYPE = torch.float32


# *** Util functions ***

def build_data_loader(dataset_class, dataset_dir, batch_size):
    """
    Builds a DataLoader object from given dataset with a SubsetRandomSampler with the full dataset

    :param dataset_class: Class of dataset to build, must take three init params: dir, img_size and num_regions
    :param dataset_dir: Path object of dataset directory (stored in hvqa format)
    :param batch_size: Size of mini-batches to read from dataset
    :return: DataLoader object which iterates over the dataset
    """

    dataset = dataset_class(dataset_dir, IMG_SIZE, NUM_REGIONS)
    num_samples = len(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(num_samples)))
    return loader
