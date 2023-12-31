# Generating Empty Aggregator to be loaded 
from transfer_attacks.Args import *
from transfer_attacks.TA_utils import dummy_aggregator
from transfer_attacks.Custom_Dataloader import *
from transfer_attacks.Transferer import *
from transfer_attacks.Params import *


import numpy as np
import copy
import torch

# setting = 'FedAvg'

def set_args(setting, num_user):

    if setting == 'FedEM':
        nL = 3
    else:
        nL = 1

    # Manually set argument parameters
    args_ = Args()
    args_.experiment = "cifar10"
    args_.method = setting
    args_.decentralized = False
    args_.sampling_rate = 1.0
    args_.input_dimension = None
    args_.output_dimension = None
    args_.n_learners= nL
    args_.n_rounds = 10
    args_.bz = 128
    args_.local_steps = 1
    args_.lr_lambda = 0
    args_.lr =0.03
    args_.lr_scheduler = 'multi_step'
    args_.log_freq = 10
    args_.device = 'cuda'
    args_.optimizer = 'sgd'
    args_.mu = 0
    args_.communication_probability = 0.1
    args_.q = 1
    args_.locally_tune_clients = False
    args_.seed = 1234
    args_.verbose = 0
    args_.save_path = 'weights/cifar/dummy/'
    args_.validation = False

    # Generate the dummy values here
    aggregator, clients = dummy_aggregator(args_, num_user=40)
    return aggregator, clients, args_

def import_model_weights(num_models, setting, save_path, aggregator, args_):

    # Import Model Weights
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    args_.save_path = save_path
    weight_path = save_path + "train_client_weights.npy"
    weights = np.load(weight_path)


    if setting == 'local':
        
        aggregator.load_state(args_.save_path)
        model_weights = []
        
        for i in range(num_models):
            model_weights += [weights[i]]

        # Generate the weights to test on as linear combinations of the model_weights
        models_test = []

        for i in range(num_models):
            new_model = copy.deepcopy(aggregator.clients[i].learners_ensemble.learners[0].model)
            new_model.eval()
            models_test += [new_model]

    elif setting == 'FedAvg':
        
        aggregator.load_state(args_.save_path)
        
        # This is where the models are stored -- one for each mixture --> learner.model for nn
        hypotheses = aggregator.global_learners_ensemble.learners

        # obtain the state dict for each of the weights 
        weights_h = []

        for h in hypotheses:
            weights_h += [h.model.state_dict()]
        
        # Set model weights
        model_weights = []

        for i in range(num_models):
            model_weights += [weights[i]]

        # Generate the weights to test on as linear combinations of the model_weights
        models_test = []

        for (w0) in model_weights:
            # first make the model with empty weights
            new_model = copy.deepcopy(hypotheses[0].model)
            new_model.eval()
            new_weight_dict = copy.deepcopy(weights_h[0])
            for key in weights_h[0]:
                new_weight_dict[key] = w0[0]*weights_h[0][key] 
            new_model.load_state_dict(new_weight_dict)
            models_test += [new_model]

    elif setting == 'FedEM':
        
        aggregator.load_state(args_.save_path)
        
        # This is where the models are stored -- one for each mixture --> learner.model for nn
        hypotheses = aggregator.global_learners_ensemble.learners

        # obtain the state dict for each of the weights 
        weights_h = []

        for h in hypotheses:
            weights_h += [h.model.state_dict()]

        # Set model weights
        model_weights = []

        for i in range(num_models):
            model_weights += [weights[i]]

        # Generate the weights to test on as linear combinations of the model_weights
        models_test = []

        for (w0,w1,w2) in model_weights:
            # first make the model with empty weights
            new_model = copy.deepcopy(hypotheses[0].model)
            new_model.eval()
            new_weight_dict = copy.deepcopy(weights_h[0])
            for key in weights_h[0]:
                new_weight_dict[key] = w0*weights_h[0][key] + w1*weights_h[1][key] + w2*weights_h[2][key]
            new_model.load_state_dict(new_weight_dict)
            models_test += [new_model]

    return models_test
    

def get_aggregate_dataloader(clients):
    # Compiling Dataset from Clients
    # Combine Validation Data across all clients as test
    data_x = []
    data_y = []

    for i in range(len(clients)):
        daniloader = clients[i].test_iterator
        for (x,y,idx) in daniloader.dataset:
            data_x.append(x)
            data_y.append(y)

    data_x = torch.stack(data_x)
    try:
        data_y = torch.stack(data_y)        
    except:
        data_y = torch.FloatTensor(data_y) 
        
    return Custom_Dataloader(data_x, data_y)
    
def generate_logs_adv(num_models):

    # Here we will make a dictionary that will hold results
    logs_adv = []

    for i in range(num_models):
        adv_dict = {}
        adv_dict['orig_acc_transfers'] = None
        adv_dict['orig_similarities'] = None
        adv_dict['adv_acc_transfers'] = None
        adv_dict['adv_similarities_target'] = None
        adv_dict['adv_similarities_untarget'] = None
        adv_dict['adv_target'] = None
        adv_dict['adv_miss'] = None
        adv_dict['metric_alignment'] = None
        adv_dict['ib_distance_legit'] = None
        adv_dict['ib_distance_adv'] = None

        logs_adv += [adv_dict]

    return logs_adv

def get_metric_list(metric_name, logs_adv, victim_idxs):

    metrics = ['orig_acc_transfers', 'orig_similarities', 'adv_acc_transfers', 'adv_similarities_target',
               'adv_similarities_untarget', 'adv_target', 'adv_miss']

    if metric_name not in metrics:
        raise ValueError(f"Invalid metric name. Choose from: {', '.join(metrics)}")

    metric_idx = metrics.index(metric_name)

    metric_list = np.zeros([len(victim_idxs), len(victim_idxs)])

    for adv_idx in range(len(victim_idxs)):
        for victim in range(len(victim_idxs)):
            metric_list[adv_idx, victim] = logs_adv[victim_idxs[adv_idx]][metrics[metric_idx]][victim_idxs[victim]].data.tolist()

    return metric_list

def cross_attack(logs_adv, victim_idxs, dataloader, models_test, custom_batch_size = 500, eps = 4.5):

    for adv_idx in victim_idxs:
        print("\t Adv idx:", adv_idx)
                
        batch_size = min(custom_batch_size, dataloader.y_data.shape[0])
        
        t1 = Transferer(models_list=models_test, dataloader=dataloader)
        t1.generate_victims(victim_idxs)
        
        # Perform Attacks Targeted
        t1.atk_params = PGD_Params()
        t1.atk_params.set_params(batch_size=batch_size, iteration = 10,
                    target = 3, x_val_min = torch.min(dataloader.x_data), 
                    x_val_max = torch.max(dataloader.x_data),
                    step_size = 0.01, step_norm = "inf", eps = eps, eps_norm = 2)
        
        
        
        t1.generate_advNN(adv_idx)
        t1.generate_xadv(atk_type = "pgd")
        t1.send_to_victims(victim_idxs)

        # Log Performance
        logs_adv[adv_idx]['orig_acc_transfers'] = copy.deepcopy(t1.orig_acc_transfers)
        logs_adv[adv_idx]['orig_similarities'] = copy.deepcopy(t1.orig_similarities)
        logs_adv[adv_idx]['adv_acc_transfers'] = copy.deepcopy(t1.adv_acc_transfers)
        logs_adv[adv_idx]['adv_similarities_target'] = copy.deepcopy(t1.adv_similarities)        
        logs_adv[adv_idx]['adv_target'] = copy.deepcopy(t1.adv_target_hit)

        # Miss attack Untargeted
        t1.atk_params.set_params(batch_size=batch_size, iteration = 10,
                    target = -1, x_val_min = torch.min(dataloader.x_data), 
                    x_val_max = torch.max(dataloader.x_data),
                    step_size = 0.01, step_norm = "inf", eps = eps, eps_norm = 2)
        t1.generate_xadv(atk_type = "pgd")
        t1.send_to_victims(victim_idxs)
        logs_adv[adv_idx]['adv_miss'] = copy.deepcopy(t1.adv_acc_transfers)
        logs_adv[adv_idx]['adv_similarities_untarget'] = copy.deepcopy(t1.adv_similarities)