import torch
from torchvision.transforms import Normalize

def unnormalize_adv(normed, dataset_name):

    if dataset_name == "cifar10" or dataset_name == "cifar100":
        return unnormalize_cifar10(normed)
    elif dataset_name == "femnist":
        return unnormalize_femnist(normed)
    # elif dataset_name == "celeba":
    #     return unnormalize_celeba(normed)

    return normed

# CIFAR10 dataset unnormalize as it comes out of the iter
def unnormalize_cifar10(normed):

    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.201])

    unnormalize = Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    a = unnormalize(normed)
    a = a.transpose(0,1)
    a = a.transpose(1,2)
    a = a * 255
    b = a.clone().detach().requires_grad_(True).type(torch.uint8)
    # b = a.clone().detach().type(torch.uint8)
    
    return b

# Celeba dataset unnormalize as it comes out of the iter
def unnormalize_celeba(normed):

    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.201])

    unnormalize = Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    a = unnormalize(normed)
    a = a * 255
    b = a.clone().detach().requires_grad_(True).type(torch.uint8)
    
    return b


def unnormalize_femnist(normed):
    mean = torch.tensor([0.1307])
    std = torch.tensor([0.3081])
    
    unnormalize = Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    a = unnormalize(normed)
    b = a.clone().detach().requires_grad_(True)
    return b