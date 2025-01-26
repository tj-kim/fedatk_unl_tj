import os
import pickle
import string

import torch
from torchvision.datasets import CIFAR10, CIFAR100, EMNIST, MNIST, CelebA
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import Dataset, ConcatDataset, Subset

import numpy as np
from PIL import Image
import gc


class TabularDataset(Dataset):
    """
    Constructs a torch.utils.Dataset object from a pickle file;
    expects pickle file stores tuples of the form (x, y) where x is vector and y is a scalar

    Attributes
    ----------
    data: iterable of tuples (x, y)

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path):
        """
        :param path: path to .pkl file
        """
        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.int64), idx


class SubFEMNIST(Dataset):
    """
    Constructs a subset of FEMNIST dataset corresponding to one client;
    Initialized with the path to a `.pt` file;
    `.pt` file is expected to hold a tuple of tensors (data, targets) storing the images and there corresponding labels.

    Attributes
    ----------
    transform
    data: iterable of integers
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path):
        self.transform = Compose([
            ToTensor(),
            Normalize((0.1307,), (0.3081,))
        ])

        self.data, self.targets = torch.load(path)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        try:
            img = np.uint8(img.numpy() * 255)
        except:
            img = np.uint8(img.detach().numpy()*255)
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class SubEMNIST(Dataset):
    """
    Constructs a subset of EMNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path, emnist_data=None, emnist_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform =\
                Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                ])

        if emnist_data is None or emnist_targets is None:
            self.data, self.targets = get_emnist()
        else:
            self.data, self.targets = emnist_data, emnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index


class SubCIFAR10(Dataset):
    """
    Constructs a subset of CIFAR10 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path, cifar10_data=None, cifar10_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param cifar10_data: Cifar-10 dataset inputs stored as torch.tensor
        :param cifar10_targets: Cifar-10 dataset labels stored as torch.tensor
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])
 
        if cifar10_data is None or cifar10_targets is None:
            self.data, self.targets = get_cifar10()
        else:
            self.data, self.targets = cifar10_data, cifar10_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index


class SubCIFAR100(Dataset):
    """
    Constructs a subset of CIFAR100 dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """
    def __init__(self, path, cifar100_data=None, cifar100_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices:
        :param cifar100_data: CIFAR-100 dataset inputs
        :param cifar100_targets: CIFAR-100 dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform = \
                Compose([
                    ToTensor(),
                    Normalize(
                        (0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010)
                    )
                ])

        if cifar100_data is None or cifar100_targets is None:
            self.data, self.targets = get_cifar100()

        else:
            self.data, self.targets = cifar100_data, cifar100_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        img = Image.fromarray(img.numpy())

        if self.transform is not None:
            img = self.transform(img)

        target = target

        return img, target, index

    
class SubMNIST(Dataset):
    """
    Constructs a subset of EMNIST dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path, mnist_data=None, mnist_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param emnist_data: EMNIST dataset inputs
        :param emnist_targets: EMNIST dataset labels
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)

        if transform is None:
            self.transform =\
                Compose([
                    ToTensor(),
                    Normalize((0.1307,), (0.3081,))
                ])

        if mnist_data is None or mnist_targets is None:
            self.data, self.targets = get_mnist()
        else:
            self.data, self.targets = mnist_data, mnist_targets

        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index
    
# class SubCeleba(Dataset):
#     """
#     Constructs a subset of celeba dataset from a pickle file;
#     expects pickle file to store list of indices

#     Attributes
#     ----------
#     indices: iterable of integers
#     transform
#     data
#     targets

#     Methods
#     -------
#     __init__
#     __len__
#     __getitem__
#     """
#     def __init__(self, path, celeba_data=None, celeba_targets=None, transform=None):
#         """
#         :param path: path to .pkl file; expected to store list of indices:
#         :param celeba_data: celeba-16 dataset inputs
#         :param celeba_targets: celeba-16 dataset labels
#         :param transform:
#         """
#         with open(path, "rb") as f:
#             self.indices = pickle.load(f)

#         self.transform = None # Transform occurs in get_celeba to reduce Gpu memory load


#         if celeba_data is None or celeba_targets is None:
#             self.data, self.targets = get_celeba()

#         else:
#             self.data, self.targets = celeba_data, celeba_targets

#         self.data = self.data[self.indices]
#         self.targets = self.targets[self.indices]

#     def __len__(self):
#         return self.data.size(0)

#     def __getitem__(self, index):
#         img, target = self.data[index], self.targets[index]

#         img = Image.fromarray(img.numpy())

#         if self.transform is not None:
#             img = self.transform(img)

#         target = target

#         return img, target, index
    
class SubCelebA(Dataset):
    """
    Constructs a subset of CelebA dataset from a pickle file;
    expects pickle file to store list of indices

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path, celeba_data=None, celeba_targets=None, transform=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param celeba_data: Concatenated train_test data
        :param transform:
        """
        with open(path, "rb") as f:
            self.indices = pickle.load(f)
            
        if celeba_data is None:
            self.data, self.targets = get_celeba()
        else:
            self.data, self.targets = celeba_data, celeba_targets
        
        self.data = self.data[self.indices]
        self.targets = self.targets[self.indices]
#         self.img = []
#         self.target = []
#         for idx, (im, tar) in enumerate(self.subset_data):
#             self.img.append(np.array(im))
#             self.target.append(tar)
#         Cant Load all the data in memory

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        return img, int(target), index

class SubFakeNewsNetwork(Dataset):
    """
    Constructs a subset of Fake news network dataset from a pickle file;
    expects pickle file to store list of [x, y]
    x - list of values that becomes embedding
    y - integer, 0 - False, 1 - True

    Attributes
    ----------
    indices: iterable of integers
    transform
    data
    targets

    Methods
    -------
    __init__
    __len__
    __getitem__
    """

    def __init__(self, path, data=None, targets=None):
        """
        :param path: path to .pkl file; expected to store list of indices
        :param celeba_data: Concatenated train_test data
        :param transform:
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.indices = None 
            
        self.data = torch.tensor(np.array([np.array(point[0]) for point in data])).unsqueeze(1) # adding channels (# points, 1, embedding length)
        self.targets = torch.tensor(np.array([point[1] for point in data]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        return img, int(target), index


class CharacterDataset(Dataset):
    def __init__(self, file_path, chunk_len):
        """
        Dataset for next character prediction, each sample represents an input sequence of characters
         and a target sequence of characters representing to next sequence of the input
        :param file_path: path to .txt file containing the training corpus
        :param chunk_len: (int) the length of the input and target sequences
        """
        self.all_characters = string.printable
        self.vocab_size = len(self.all_characters)
        self.n_characters = len(self.all_characters)
        self.chunk_len = chunk_len

        with open(file_path, 'r') as f:
            self.text = f.read()

        self.tokenized_text = torch.zeros(len(self.text), dtype=torch.long)

        self.inputs = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)
        self.targets = torch.zeros(self.__len__(), self.chunk_len, dtype=torch.long)

        self.__build_mapping()
        self.__tokenize()
        self.__preprocess_data()

    def __tokenize(self):
        for ii, char in enumerate(self.text):
            self.tokenized_text[ii] = self.char2idx[char]

    def __build_mapping(self):
        self.char2idx = dict()
        for ii, char in enumerate(self.all_characters):
            self.char2idx[char] = ii

    def __preprocess_data(self):
        for idx in range(self.__len__()):
            self.inputs[idx] = self.tokenized_text[idx:idx+self.chunk_len]
            self.targets[idx] = self.tokenized_text[idx+1:idx+self.chunk_len+1]

    def __len__(self):
        return max(0, len(self.text) - self.chunk_len)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx], idx


def get_emnist():
    """
    gets full (both train and test) EMNIST dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        emnist_data, emnist_targets
    """
    emnist_path = os.path.join("data", "emnist", "raw_data")
    assert os.path.isdir(emnist_path), "Download EMNIST dataset!!"

    emnist_train =\
        EMNIST(
            root=emnist_path,
            split="byclass",
            download=True,
            train=True
        )

    emnist_test =\
        EMNIST(
            root=emnist_path,
            split="byclass",
            download=True,
            train=True
        )

    emnist_data =\
        torch.cat([
            emnist_train.data,
            emnist_test.data
        ])

    emnist_targets =\
        torch.cat([
            emnist_train.targets,
            emnist_test.targets
        ])

    return emnist_data, emnist_targets


def get_cifar10():
    """
    gets full (both train and test) CIFAR10 dataset inputs and labels;
    the dataset should be first downloaded (see data/emnist/README.md)
    :return:
        cifar10_data, cifar10_targets
    """
    cifar10_path = os.path.join("data", "cifar10", "raw_data")
    assert os.path.isdir(cifar10_path), "Download cifar10 dataset!!"

    cifar10_train =\
        CIFAR10(
            root=cifar10_path,
            train=True, download=False
        )

    cifar10_test =\
        CIFAR10(
            root=cifar10_path,
            train=False,
            download=False)

    cifar10_data = \
        torch.cat([
            torch.tensor(cifar10_train.data),
            torch.tensor(cifar10_test.data)
        ])

    cifar10_targets = \
        torch.cat([
            torch.tensor(cifar10_train.targets),
            torch.tensor(cifar10_test.targets)
        ])

    return cifar10_data, cifar10_targets


def get_cifar100():
    """
    gets full (both train and test) CIFAR100 dataset inputs and labels;
    the dataset should be first downloaded (see data/cifar100/README.md)
    :return:
        cifar100_data, cifar100_targets
    """
    cifar100_path = os.path.join("data", "cifar100", "raw_data")
    assert os.path.isdir(cifar100_path), "Download cifar10 dataset!!"

    cifar100_train =\
        CIFAR100(
            root=cifar100_path,
            train=True, download=False
        )

    cifar100_test =\
        CIFAR100(
            root=cifar100_path,
            train=False,
            download=False)

    cifar100_data = \
        torch.cat([
            torch.tensor(cifar100_train.data),
            torch.tensor(cifar100_test.data)
        ])

    cifar100_targets = \
        torch.cat([
            torch.tensor(cifar100_train.targets),
            torch.tensor(cifar100_test.targets)
        ])

    return cifar100_data, cifar100_targets

def get_mnist():
    mnist_path = os.path.join("data", "mnist", "raw_data")
    assert os.path.isdir(mnist_path), "Download mnist dataset!!"
    
    mnist_train =\
        MNIST(
            root= mnist_path,
            train=True, download=False
        )

    mnist_test =\
        MNIST(
            root=mnist_path,
            train=False,
            download=False)

    mnist_data = \
        torch.cat([
            torch.tensor(mnist_train.data),
            torch.tensor(mnist_test.data)
        ])

    mnist_targets = \
        torch.cat([
            torch.tensor(mnist_train.targets),
            torch.tensor(mnist_test.targets)
        ])

    return mnist_data, mnist_targets

# def get_celeba():
#     """
#     gets full (both train and test) celeba dataset inputs and labels;
#     the dataset should be first downloaded (see data/emnist/README.md)
#     :return:
#         celeba_data, celeba_targets
#     """
#     celeba_path = os.path.join("data", "celeba", "raw_data")
#     assert os.path.isdir(celeba_path), "Download celeba dataset!!"

#     transform = Compose([
#             Resize((40, 40)),
#             ToTensor(),
#             Normalize(
#                         (0.4914, 0.4822, 0.4465),
#                         (0.2023, 0.1994, 0.2010)
#                     )])

#     celeba_train =\
#         CelebA(
#             root=celeba_path,
#             split='train', download=False,
#             transform = transform,
#             target_transform=lambda x: transform_target(x, required_labels = [31, 20, 15, 35])
#         )

#     celeba_test =\
#         CelebA(
#             root=celeba_path,
#             split='test',
#             download=False,
#             transform = transform,
#             target_transform=lambda x: transform_target(x, required_labels = [31, 20, 15, 35]))

#     celeba_data = \
#         torch.cat([
#             torch.tensor(celeba_train.data),
#             torch.tensor(celeba_test.data)
#         ])

#     celeba_targets = \
#         torch.cat([
#             torch.tensor(celeba_train.targets),
#             torch.tensor(celeba_test.targets)
#         ])

#     return celeba_data, celeba_targets

# def get_celeba():
#     celeba_path = os.path.join("data", "celeba", "raw_data")
#     assert os.path.isdir(celeba_path), "Download celeba dataset!!"
    
#     transform =\
#                 Compose([
#                     #Resize((45, 55)),
#                     ToTensor(),
#                     Normalize(
#                         (0.4914, 0.4822, 0.4465),
#                         (0.2023, 0.1994, 0.2010)
#                     )
#                 ])
    
#     celeba_train =\
#         CelebA(
#             root= celeba_path,
#             split='train', download=False,
#             transform = transform,
#             target_transform=lambda x: transform_target(x, required_labels = [31, 20, 15, 35]) # Smiling, Male, Eyeglasses, Wearing Hat
#         )
    
#     train_idx = np.load('data/celeba/train_idx.npy', allow_pickle = True)
#     test_idx = np.load('data/celeba/test_idx.npy', allow_pickle = True)

#     celeba_test =\
#         CelebA(
#             root=celeba_path,
#             split='test',
#             download=False,
#             transform = transform,
#             target_transform=lambda x: transform_target(x, required_labels = [31, 20, 15, 35])
#     )
    
#     celeba_train = Subset(celeba_train, train_idx)
#     celeba_test = Subset(celeba_test, test_idx)
    
#     celeba_data_X = []
#     celeba_data_y = []
    
#     for idx, data in enumerate(celeba_train):
#         celeba_data_X.append(data[0])
#         celeba_data_y.append(data[1])
    
#     for idx, data in enumerate(celeba_test):
#         celeba_data_X.append(data[0])
#         celeba_data_y.append(data[1])
        
#     return torch.stack(celeba_data_X), torch.Tensor(celeba_data_y)

def get_celeba():
    # Load and combine all batches
    print("get_celeba - Combining train batches... float16 version")
    x_train, y_train = load_numpy_batches("celeba_pickle/train_batches")
    x_train_tensor = torch.from_numpy(x_train)
    y_train_tensor = torch.from_numpy(y_train)

    print("get_celeba -Combining test batches... float16 version")
    x_test, y_test = load_numpy_batches("celeba_pickle/test_batches")

    # Convert to PyTorch tensors
    x = torch.from_numpy(np.concatenate([x_train, x_test], axis=0))
    y = torch.from_numpy(np.concatenate([y_train, y_test], axis=0))

    del x_train, y_train, x_test, y_test
    gc.collect()
    torch.cuda.empty_cache()


    return x, y

# Celeba tool
def transform_target(target, required_labels=[31]):
    target_str = ''
    for label in required_labels:
        target_str += str(int(target[label]))
    return int(target_str, 2)

def load_numpy_batches(file_dir):
    # Load all batches from the directory and concatenate them
    data_list, label_list = [], []
    for file_name in sorted(os.listdir(file_dir)):  # Sort to maintain batch order
        file_path = os.path.join(file_dir, file_name)
        with open(file_path, "rb") as f:
            data, labels = pickle.load(f)
            data_list.append(data)
            label_list.append(labels)
    return np.concatenate(data_list, axis=0), np.concatenate(label_list, axis=0)