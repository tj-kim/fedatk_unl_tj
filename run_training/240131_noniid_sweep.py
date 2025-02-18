"""Run Experiment

This script allows to run one federated learning experiment; the experiment name, the method and the
number of clients/tasks should be precised along side with the hyper-parameters of the experiment.

The results of the experiment (i.e., training logs) are written to ./logs/ folder.

This file can also be imported as a module and contains the following function:

    * run_experiment - runs one experiments given its arguments
"""

# Import General Libraries
import os
import argparse
import torch
import copy
import pickle
import random
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import numba 

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.utils import *
from utils.constants import *
from utils.args import *
from run_experiment import * 
from models import *


# Import Transfer Attack
from transfer_attacks.Personalized_NN import *
from transfer_attacks.Params import *
from transfer_attacks.Transferer import *
from transfer_attacks.Args import *
from transfer_attacks.TA_utils import *



if __name__ == "__main__":
    
    os.chdir(parent_dir) # As we are in a folder

    exp_names =['tm_iid', 'tm_niid', 'med_iid', 'med_niid']
    iid_mode = [True, False, True, False]
    agg_op_list =  ["trimmed_mean", "trimmed_mean", "median", "median"]
    exp_method = ['FedAvg_adv','FedAvg_adv','FedAvg_adv','FedAvg_adv']
    save_folder = 'weights/cifar10/250123_icml25/'
    agg_tm_rate = 0.1

    exp_num_learners = 1
    exp_lr = 0.01
    exp_name = 'cifar10'
    num_rounds = 200
    num_clients = 40
    
        
    for itt in range(len(exp_names)):
        
        print("running trial:", itt+1, "of", len(exp_names), ":", exp_names[itt])
        
        # Manually set argument parameters
        args_ = Args()
        args_.experiment = exp_name
        args_.method = exp_method[itt]
        args_.decentralized = False
        args_.sampling_rate = 1.0
        args_.input_dimension = None
        args_.output_dimension = None
        args_.n_learners= exp_num_learners
        args_.n_rounds = num_rounds
        args_.bz = 128
        args_.local_steps = 1
        args_.lr_lambda = 0
        args_.lr = exp_lr
        args_.lr_scheduler = 'multi_step'
        args_.log_freq = 10
        args_.device = 'cuda'
        args_.optimizer = 'sgd'
        args_.mu = 0
        args_.communication_probability = 0.1
        args_.q = 1
        args_.locally_tune_clients = False
        args_.seed = 1234
        args_.verbose = 1
        args_.save_path = save_folder + exp_names[itt]
        args_.validation = False
        args_.save_freq = 10

        # Other Argument Parameters
        Q = 10 # update per round
        G = 0.4
        S = 0.05 # Threshold
        step_size = 0.01
        K = 10
        eps = 4.5

        # Randomized Parameters
        # Ru = np.random.uniform(0, 0.5, size=num_clients)
        Ru = np.ones(num_clients)
        
        # Generate the dummy values here
        aggregator, clients = dummy_aggregator(args_, num_clients)
        aggregator.aggregation_op = agg_op_list[itt]
        aggregator.tm_rate = agg_tm_rate
        
        
        # Set attack parameters
        if exp_method[itt] == 'FedAvg_adv':
            x_min = torch.min(clients[0].adv_nn.dataloader.x_data)
            x_max = torch.max(clients[0].adv_nn.dataloader.x_data)
            atk_params = PGD_Params()
            atk_params.set_params(batch_size=1, iteration = K,
                            target = -1, x_val_min = x_min, x_val_max = x_max,
                            step_size = 0.01, step_norm = "inf", eps = eps, eps_norm = 2)

        # Obtain the central controller decision making variables (static)
        num_h = args_.n_learners= 3
        Du = np.zeros(len(clients))

        for i in range(len(clients)):
            num_data = clients[i].train_iterator.dataset.targets.shape[0]
            Du[i] = num_data
        D = np.sum(Du) # Total number of data points


        # Train the model
        print("Training..")
        pbar = tqdm(total=args_.n_rounds)
        current_round = 0
        while current_round <= args_.n_rounds:

            if exp_method[itt] == 'FedAvg_adv':
                # If statement catching every Q rounds -- update dataset
                if  current_round != 0 and current_round%Q == 0: # 
                    # print("Round:", current_round, "Calculation Adv")
                    # Obtaining hypothesis information
                    # Solve for adversarial ratio at every client
                    # Fu = solve_proportions(G, num_clients, num_h, Du, Whu, S, Ru, step_size)
                    if iid_mode[itt]:
                        Fu = np.ones(num_clients) * G
                    else:
                        Fu = np.zeros(num_clients)
                        Fu[0:16] = 1 # 15 for cifar 10 

                    # Assign proportion and attack params
                    for i in range(len(clients)):
                        if Fu[i] > 0:
                            aggregator.clients[i].set_adv_params(Fu[i], atk_params)
                            aggregator.clients[i].update_advnn()
                            aggregator.clients[i].assign_advdataset()

            aggregator.mix()
            
            # Save more often the intermediate NN
            if current_round% args_.save_freq == 0:
                if "save_path" in args_:
                    save_root = os.path.join(args_.save_path)

                    os.makedirs(save_root, exist_ok=True)

            if aggregator.c_round != current_round:
                pbar.update(1)
                current_round = aggregator.c_round

        if "save_path" in args_:
            save_root = os.path.join(args_.save_path)

            os.makedirs(save_root, exist_ok=True)
            aggregator.save_state(save_root)
            
        del args_, aggregator, clients
        torch.cuda.empty_cache()
            