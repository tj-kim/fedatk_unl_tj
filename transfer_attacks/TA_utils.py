import argparse, torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from pathlib import Path

# Import General Libraries
import os
import argparse
import copy
import pickle
import random

# Import FedEM based Libraries
from utils.utils import *
from utils.constants import *
from utils.args import *
from torch.utils.tensorboard import SummaryWriter
from run_experiment import *
from models import *

from sklearn.metrics.pairwise import cosine_similarity



class One_Hot(nn.Module):
    # from :
    # https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    def __init__(self, depth):
        super(One_Hot,self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)
    def forward(self, X_in):
        X_in = X_in.long()
        return Variable(self.ones.index_select(0,X_in.data))
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


def cuda(tensor,is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def rm_dir(dir_path, silent=True):
    p = Path(dir_path).resolve()
    if (not p.is_file()) and (not p.is_dir()) :
        print('It is not path for file nor directory :',p)
        return

    paths = list(p.iterdir())
    if (len(paths) == 0) and p.is_dir() :
        p.rmdir()
        if not silent : print('removed empty dir :',p)

    else :
        for path in paths :
            if path.is_file() :
                path.unlink()
                if not silent : print('removed file :',path)
            else:
                rm_dir(path)
        p.rmdir()
        if not silent : print('removed empty dir :',p)

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)



def dummy_aggregator(args_, num_user=80):

    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_root" in args_:
        logs_root = args_.logs_root
    else:
        logs_root = os.path.join("logs", args_to_string(args_))

    print("==> Clients initialization..")
    clients_temp = init_clients(
        args_,
        root_path=os.path.join(data_dir, "train"),
        logs_root=os.path.join(logs_root, "train"),
        client_limit = num_user, 
    )

    clients = clients_temp[:num_user]

    ### REMOVED DUE TO MEMORY LIMITS    
    # print("==> Test Clients initialization..")
    # test_clients_temp = init_clients(
    #     args_,
    #     root_path=os.path.join(data_dir, "test"),
    #     logs_root=os.path.join(logs_root, "test"),
    #     client_limit = num_user
    # )
    
    # test_clients = test_clients_temp[:num_user]
    test_clients = clients

    logs_path = os.path.join(logs_root, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_root, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu
    )


    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]

    aggregator =\
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            seed=args_.seed
        )

    return aggregator, clients

def copy_aggregator(aggregator_og, args_):

    num_user = len(aggregator_og.clients)

    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_root" in args_:
        logs_root = args_.logs_root
    else:
        logs_root = os.path.join("logs", args_to_string(args_))

    train_iterators = []
    val_iterators = []
    test_iterators = []
    client_learners_ensemble = copy.deepcopy(aggregator_og.clients[0].learners_ensemble)

    for client in aggregator_og.clients:
        train_iterators += [copy.deepcopy(client.train_iterator)]
        val_iterators += [copy.deepcopy(client.val_iterator)]
        test_iterators += [copy.deepcopy(client.test_iterator)]

    clients_temp = init_clients_rip_iterators(
        args_,
        train_iterators, 
        val_iterators, 
        test_iterators,
        client_learners_ensemble,
        logs_root=os.path.join(logs_root, "train"),
        client_limit = num_user
    )

    clients = clients_temp[:num_user]
    test_clients = clients

    logs_path = os.path.join(logs_root, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_root, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    if args_.decentralized:
        aggregator_type = 'decentralized'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]

    aggregator =\
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=copy.deepcopy(aggregator_og.global_learners_ensemble),
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            seed=args_.seed
        )

    aggregator.update_clients()

    return aggregator

def init_clients_rip_iterators(args_, train_iterators, val_iterators, test_iterators, learners_ensemble, 
                         logs_root, client_limit = None):
    """
    initialize clients from data folders
    :param args_:
    :param root_path: path to directory containing data folders
    :param logs_root: path to logs root
    :return: List[Client]
    """

    clients_ = []
    for task_id, (train_iterator, val_iterator, test_iterator) in \
            enumerate(tqdm(zip(train_iterators, val_iterators, test_iterators), total=len(train_iterators))):

        if train_iterator is None or test_iterator is None:
            continue

        logs_path = os.path.join(logs_root, "task_{}".format(task_id))
        os.makedirs(logs_path, exist_ok=True)
        logger = SummaryWriter(logs_path)

        client = get_client(
            client_type=CLIENT_TYPE[args_.method],
            learners_ensemble=learners_ensemble,
            q=args_.q,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=args_.local_steps,
            tune_locally=args_.locally_tune_clients
        )

        clients_.append(client)

    return clients_

def dummy_aggregator_distmec(args_, num_user=80):

    torch.manual_seed(args_.seed)

    data_dir = get_data_dir(args_.experiment)

    if "logs_root" in args_:
        logs_root = args_.logs_root
    else:
        logs_root = os.path.join("logs", args_to_string(args_))

    print("==> Clients initialization..")
    clients_temp = init_clients(
        args_,
        root_path=os.path.join(data_dir, "train"),
        logs_root=os.path.join(logs_root, "train")
    )

    clients = clients_temp[:num_user]
    
    print("==> Test Clients initialization..")
    test_clients_temp = init_clients(
        args_,
        root_path=os.path.join(data_dir, "test"),
        logs_root=os.path.join(logs_root, "test")
    )
    
    test_clients = test_clients_temp[:num_user]

    logs_path = os.path.join(logs_root, "train", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_train_logger = SummaryWriter(logs_path)

    logs_path = os.path.join(logs_root, "test", "global")
    os.makedirs(logs_path, exist_ok=True)
    global_test_logger = SummaryWriter(logs_path)

    global_learners_ensemble = \
        get_learners_ensemble(
            n_learners=args_.n_learners,
            name=args_.experiment,
            device=args_.device,
            optimizer_name=args_.optimizer,
            scheduler_name=args_.lr_scheduler,
            initial_lr=args_.lr,
            input_dim=args_.input_dimension,
            output_dim=args_.output_dimension,
            n_rounds=args_.n_rounds,
            seed=args_.seed,
            mu=args_.mu
    )


    if args_.decentralized:
        aggregator_type = 'distmec'
    else:
        aggregator_type = AGGREGATOR_TYPE[args_.method]

    aggregator =\
        get_aggregator(
            aggregator_type=aggregator_type,
            clients=clients,
            global_learners_ensemble=global_learners_ensemble,
            lr_lambda=args_.lr_lambda,
            lr=args_.lr,
            q=args_.q,
            mu=args_.mu,
            communication_probability=args_.communication_probability,
            sampling_rate=args_.sampling_rate,
            log_freq=args_.log_freq,
            global_train_logger=global_train_logger,
            global_test_logger=global_test_logger,
            test_clients=test_clients,
            verbose=args_.verbose,
            seed=args_.seed
        )

    return aggregator, clients

# ADV functions

# Solve for Fu for all users
def solve_proportions(G, N, num_h, Du, Whu, S, Ru, step_size):
    """
    Inputs:
    - G - Desired proportion of adv data points
    - N - Number of users in the system
    - num_h - Number of mixtures/hypotheses (FedEM)
    - Du - Number of data points at user U
    - Whu - Weight of each hypothis at user U
    - S - Threshold for objective function to fall below
    - Ru - Resource limits at each user (proportion)
    - step_size - For sweeping Fu
    Output:
    - Fu - proportion of adv data for each client
    """
    
    # finalize information needed to solve problem
    Wh = np.sum(Whu,axis=0)/N
    D = np.sum(Du)

    Fu = np.ones_like(Ru) * G

    # Step 1. Initial filter out all users with less resource constraints
    A = np.where(Fu>Ru)[0]
    B = np.where(Fu<Ru)[0]
    Fu[A] = Ru[A]

    # Step 2. Select users at random and change proportion, check objective 
    np.random.shuffle(B)
    for i in B:
        curr_obj = calc_prop_objective(G, num_h, Du, Whu, Fu)
        while Fu[i] + step_size < Ru[i]:
            Fu_temp = copy.deepcopy(Fu)
            Fu_temp[i] = Fu[i] + step_size
            new_obj = calc_prop_objective(G, num_h, Du, Whu, Fu_temp)
            if new_obj <= S:
                Fu = Fu_temp
                break
            elif new_obj < curr_obj:
                Fu = Fu_temp
                curr_obj = new_obj
            else: break
                
    return Fu

# Solve for Fu for all users
def solve_proportions_dummy(G, N, num_h, Du, Whu, S, Ru, step_size):
    """
    Inputs:
    - G - Desired proportion of adv data points
    - N - Number of users in the system
    - num_h - Number of mixtures/hypotheses (FedEM)
    - Du - Number of data points at user U
    - Whu - Weight of each hypothis at user U
    - S - Threshold for objective function to fall below
    - Ru - Resource limits at each user (proportion)
    - step_size - For sweeping Fu
    Output:
    - Fu - proportion of adv data for each client
    """
    
    # finalize information needed to solve problem
    Wh = np.sum(Whu,axis=0)/N
    D = np.sum(Du)

    Fu = np.ones_like(Ru) * G

    # Step 1. Initial filter out all users with less resource constraints
    A = np.where(Fu>Ru)[0]
    B = np.where(Fu<Ru)[0]
    Fu[A] = Ru[A]
                
    return Fu

def calc_prop_objective(G, num_h, Du, Whu, Fu):
# Calculate objective function value for attaining global adv data proportion
    N = Whu.shape[0]
    Wh = np.sum(Whu,axis=0)/N
    obj = 0
    D = np.sum(Du)
    for n in range(num_h):    
        obj += np.abs(np.sum(Fu * Du * Whu[:,n])- G * D * Wh[n]) * 1/D
    return obj

# Perform np.mean without the diagonal
def avg_nondiag(array2d):
    d1 = array2d.shape[0]
    d2 = array2d.shape[1]
    
    counter = 0
    val = 0
    
    for i1 in range(d1):
        for i2 in range(d2):
            if i1 != i2:
                counter+=1
                val += array2d[i1,i2]
    
    return val/counter

# Make a pandas table across double sweep from list
def make_metric_table(exp_list, metric, row_names, col_names, avg_diag_flag = True):
    
    num_col1 = len(exp_list)
    num_col2 = len(exp_list[0])
    num_victims = len(exp_list[0][0])
    victim_idxs = range(num_victims)
    exp_values = {}
    
    final_table = np.zeros([num_col1, num_col2])
    
    for j in range(num_col1): # Attack perturbation amount
        for k in range(num_col2): # Defense perturbation amount (Experiment)
            orig_vals = np.zeros([num_victims, num_victims])
            
            for adv_idx in range(num_victims):
                for victim in range(num_victims):
                    curr_list = exp_list[j][k]
                    orig_vals[adv_idx,victim] = curr_list[victim_idxs[adv_idx]][metric][victim_idxs[victim]].data.tolist()
            
            if avg_diag_flag:
                final_table[j,k] = avg_nondiag(orig_vals)
            else:
                final_table[j,k] = np.mean(orig_vals)
    
    df = pd.DataFrame(final_table, columns = col_names, index = row_names)
    
    return df

def load_client_data(clients, c_id, mode = 'test'):
    
    data_x = []
    data_y = []

    if mode == 'all': # load all test sets together
        for i in range(len(clients)):
            daniloader = clients[i].test_iterator
            for (x,y,idx) in daniloader.dataset:
                data_x.append(x)
                data_y.append(y)
    else:
        if mode == 'train':
            daniloader = clients[c_id].train_iterator
        elif mode == 'val':
            daniloader = clients[c_id].val_iterator
        elif mode == 'test':
            daniloader = clients[c_id].test_iterator

        for (x,y,idx) in daniloader.dataset:
            data_x.append(x)
            data_y.append(y)

    data_x = torch.stack(data_x)
    try:
        data_y = torch.stack(data_y)        
    except:
        data_y = torch.FloatTensor(data_y) 
        
    dataloader = Custom_Dataloader(data_x, data_y)
    
    return dataloader

def update_aggregator_dataset(aggregator, data_prop):
    
    for c_id in range(len(aggregator.clients)):
            dataset = aggregator.clients[c_id].train_iterator.dataset
            num_points = dataset.targets.shape[0]
            new_num = int(num_points * data_prop)
            
#             dataset.targets = dataset.targets[:new_num]
#             dataset.data = dataset.data[:new_num]
#             dataset.indices = dataset.indices[:new_num]
            
            aggregator.clients[c_id].train_iterator.dataset.targets = dataset.targets[:new_num]
            aggregator.clients[c_id].train_iterator.dataset.data = dataset.data[:new_num]
#             aggregator.clients[c_id].train_iterator.dataset.indices = dataset.indices[:new_num]
            
#             aggregator.clients[c_id].train_iterator.dataset = dataset

    return


def matrix_cosine_similarity(mat1, mat2):
    vec1 = mat1.cpu().numpy().flatten()
    vec2 = mat2.cpu().numpy().flatten()
    return cosine_similarity([vec1], [vec2])[0][0]