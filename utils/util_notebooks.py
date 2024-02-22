# Generating Empty Aggregator to be loaded 
from transfer_attacks.Args import *
from transfer_attacks.TA_utils import dummy_aggregator, load_client_data
from transfer_attacks.Custom_Dataloader import *
from transfer_attacks.Transferer import *
from transfer_attacks.Params import *


import numpy as np
import copy
import torch

# setting = 'FedAvg'

def set_args(setting, num_user, experiment = "cifar10"):

    if setting == 'FedEM':
        nL = 3
    else:
        nL = 1

    # Manually set argument parameters
    args_ = Args()
    args_.experiment = experiment
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
    args_.lr =0.01
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
    aggregator, clients = dummy_aggregator(args_, num_user)
    return aggregator, clients, args_

def import_model_weights(num_models, setting, save_path, aggregator, args_):

    # Import Model Weights
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
    args_.save_path = save_path
    weight_path = save_path + "train_client_weights.npy"
    weights = np.load(weight_path)


    if setting == 'local' or setting == 'local_adv':
        
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

    elif setting == 'FedAvg' or setting == 'FedAvg_adv':
        
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

    elif setting == 'FedEM' or setting == 'FedEM_adv':
        
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


### Below is for cosine similarity experiments
### Where gradual transition/catastrophic forgetting is performed
from sklearn.metrics.pairwise import cosine_similarity

def matrix_cosine_similarity(mat1, mat2):
    vec1 = mat1.cpu().numpy().flatten()
    vec2 = mat2.cpu().numpy().flatten()
    return cosine_similarity([vec1], [vec2])[0][0]

def get_diff_NN( model1, model2, desired_keys):

    param_model1 = model1.state_dict()
    param_model2 = model2.state_dict()

    mag_norm_122 = []
    for key in desired_keys: #params_FAT:

        diff = param_model1[key] - param_model2[key]
        l2_norm = torch.norm(diff, p=2)

        mag_norm_122 += [diff/torch.norm(diff,p=2)]
    return mag_norm_122

def diff_cosine_similarity(diff, baseline, key_length):
    values_stored = np.zeros(key_length)

    for i in range(key_length):
        values_stored[i] = matrix_cosine_similarity(diff[i], baseline[i])

    return values_stored 

def initialize_logsadv(num_models):

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

def get_adv_acc(aggregator, model, batch_size = 500):
    num_clients = len(aggregator.clients)

    # logs_adv = generate_logs_adv(num_models=num_clients)

    # Dataloader for datax
    data_x = []
    daniloader = aggregator.clients[0].val_iterator
    for (x,y,idx) in daniloader.dataset:
        data_x.append(x)

    data_x = torch.stack(data_x)
    # victim_idxs = range(num_clients)

    # Save matrix
    test_acc_save = np.zeros([num_clients])
    adv_acc_save = np.zeros([num_clients])

    for c_id in range(num_clients):
        victim_idxs = range(1) # just test against one client
        dataloader = load_client_data(clients = aggregator.clients, c_id = c_id, mode = 'test')
        batch_size = min(batch_size, dataloader.y_data.shape[0])

        t1 = Transferer(models_list = [model] * num_clients, dataloader=dataloader)
        t1.generate_victims(victim_idxs)
        t1.atk_params = PGD_Params()
        t1.atk_params.set_params(batch_size=batch_size, iteration = 10, target = -1,
                                x_val_min = torch.min(data_x), x_val_max = torch.max(data_x),
                                step_size = 0.05, step_norm = "inf", eps = 4, eps_norm = 2)
        t1.generate_advNN(c_id)
        t1.generate_xadv(atk_type="pgd")
        t1.send_to_victims(victim_idxs)
        test_acc_save[c_id] = t1.orig_acc_transfers[0]
        adv_acc_save[c_id] = t1.adv_acc_transfers[0]

    return test_acc_save, adv_acc_save 

def pull_model_from_agg(aggregator):
        
    # This is where the models are stored -- one for each mixture --> learner.model for nn
    hypotheses = aggregator.global_learners_ensemble.learners

    # obtain the state dict for each of the weights 
    weights_h = []

    for h in hypotheses:
        weights_h += [h.model.state_dict()]
    
    # first make the model with empty weights
    new_model = copy.deepcopy(hypotheses[0].model)
    return new_model

######## 
#   BELOW IS ARU INJECTION STUFF
########

# Calculate uploaded model and download to attacker clients in aggregator
# Current version working under the assumption of close to convergence (no benign client pushback)
def calc_atk_model(model_inject, model_global, keys, weight_scale, weight_scale_2):

    atk_model = copy.deepcopy(model_global)
    inject_state_dict = model_inject.state_dict(keep_vars=True)
    global_state_dict = model_global.state_dict(keep_vars=True)
    return_state_dict = atk_model.state_dict(keep_vars=True)
    total_weight = weight_scale * weight_scale_2

    for key in keys:
        diff = inject_state_dict[key].data.clone() - global_state_dict[key].data.clone()
        return_state_dict[key].data = total_weight * diff + global_state_dict[key].data.clone()

    return atk_model

# Expand aggregator.mix() function
def UNL_mix(aggregator, adv_id, model_inject, keys, weight_scale_2, dump_flag=False, aggregation_op = None, tm_beta = 0.05):
    weight_scale = 1/aggregator.clients_weights
    model_global = aggregator.global_learners_ensemble[0].model

    if aggregation_op == None:
        aggregation_op = aggregator.aggregation_op

    # Give adversarial clients boosted models and train regular clients 1 round
    benign_id = list(range(len(aggregator.clients)))
    for a_id in adv_id:
        benign_id.remove(a_id)
        temp_atk_model = calc_atk_model(model_inject, model_global, keys, weight_scale[a_id], weight_scale_2)
        aggregator.clients[a_id].learners_ensemble[0].model = copy.deepcopy(temp_atk_model)

    for c_id in benign_id:
        aggregator.clients[c_id].step()

    # Aggregate model and download
    for learner_id, learner in enumerate(aggregator.global_learners_ensemble):
        learners = [client.learners_ensemble[learner_id] for client in aggregator.clients]
        if aggregation_op is None:
            average_learners(learners, learner, weights=aggregator.clients_weights)
        elif aggregation_op == 'median':
            dump_path = (
                os.path.join(aggregator.dump_path, f"round{aggregator.c_round}_median.pkl") 
                if dump_flag
                else None
            )
            byzantine_robust_aggregate_median(
                learners, 
                learner, 
                dump_path=dump_path
            )
        elif aggregation_op == 'trimmed_mean':
            dump_path = (
                os.path.join(aggregator.dump_path, f"round{aggregator.c_round}_tm.pkl")
                if dump_flag
                else None
            )
            byzantine_robust_aggregate_tm(
                learners, 
                learner, 
                beta=tm_beta, 
                dump_path=dump_path
            )
        elif aggregation_op == 'krum':
            dump_path = (
                os.path.join(aggregator.dump_path, f"round{aggregator.c_round}_krum.pkl")
                if dump_flag
                else None
            )
            byzantine_robust_aggregate_krum(
                learners, 
                learner, 
                dump_path=dump_path
            )
        elif aggregation_op == 'krum_modelwise':
            dump_path = (
                os.path.join(aggregator.dump_path, f"round{aggregator.c_round}_krum_modelwise.pkl")
                if dump_flag
                else None
            )
            byzantine_robust_aggregate_krum_modelwise(
                1,
                learners,
                learner,
                dump_path=dump_path
            )
        else:
            raise NotImplementedError


    # assign the updated model to all clients
    aggregator.update_clients()

    aggregator.c_round += 1

    # if aggregator.c_round % aggregator.log_freq == 0:
    #     aggregator.write_logs()
    return 