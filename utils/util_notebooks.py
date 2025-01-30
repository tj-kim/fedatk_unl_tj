# Generating Empty Aggregator to be loaded 
from transfer_attacks.Args import *
from transfer_attacks.TA_utils import dummy_aggregator, load_client_data
from transfer_attacks.Custom_Dataloader import *
from transfer_attacks.Transferer import *
from transfer_attacks.Params import *
from utils.torch_utils import *

import numpy as np
import copy
import torch
import torch.nn.functional as F
import re
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

def cross_attack(logs_adv, victim_idxs, dataloader, models_test, custom_batch_size = 500, eps = 4.5, atk_steps = 10):

    for adv_idx in victim_idxs:
        print("\t Adv idx:", adv_idx)
                
        batch_size = min(custom_batch_size, dataloader.y_data.shape[0])
        
        t1 = Transferer(models_list=models_test, dataloader=dataloader)
        t1.generate_victims(victim_idxs)
        
        # Perform Attacks Targeted
        t1.atk_params = PGD_Params()
        t1.atk_params.set_params(batch_size=batch_size, iteration = atk_steps,
                    target = 1, x_val_min = torch.min(dataloader.x_data), 
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
        t1.atk_params.set_params(batch_size=batch_size, iteration = atk_steps,
                    target = -1, x_val_min = torch.min(dataloader.x_data), 
                    x_val_max = torch.max(dataloader.x_data),
                    step_size = 0.01, step_norm = "inf", eps = eps, eps_norm = 2)
        t1.generate_xadv(atk_type = "pgd")
        t1.send_to_victims(victim_idxs)
        logs_adv[adv_idx]['adv_miss'] = copy.deepcopy(t1.adv_acc_transfers)
        logs_adv[adv_idx]['adv_similarities_untarget'] = copy.deepcopy(t1.adv_similarities)


def cross_attack_target(logs_adv, victim_idxs, dataloader, models_test, 
                        target_labels, custom_batch_size=500, eps=4.5, atk_steps=10):

    for adv_idx in victim_idxs:
        print("\t Adv idx:", adv_idx)
                
        batch_size = min(custom_batch_size, dataloader.y_data.shape[0])
        
        t1 = Transferer(models_list=models_test, dataloader=dataloader)
        t1.generate_victims(victim_idxs)
        
        # Perform Attacks Targeted
        t1.atk_params = PGD_Params()
        t1.atk_params.set_params(batch_size=batch_size, iteration = atk_steps,
                    target = 3, x_val_min = torch.min(dataloader.x_data), 
                    x_val_max = torch.max(dataloader.x_data),
                    step_size = 0.01, step_norm = "inf", eps = eps, eps_norm = 2)
        
        
        
        t1.generate_advNN(adv_idx)
        t1.generate_xadv(atk_type = "pgd")
        t1.send_to_victims(victim_idxs, target_labels)

        # Log Performance
        logs_adv[adv_idx]['orig_acc_transfers'] = copy.deepcopy(t1.orig_acc_transfers)
        logs_adv[adv_idx]['orig_similarities'] = copy.deepcopy(t1.orig_similarities)
        logs_adv[adv_idx]['adv_acc_transfers'] = copy.deepcopy(t1.adv_acc_transfers)
        logs_adv[adv_idx]['adv_similarities_target'] = copy.deepcopy(t1.adv_similarities)        
        logs_adv[adv_idx]['adv_target'] = copy.deepcopy(t1.adv_target_hit)

        # Miss attack Untargeted
        t1.atk_params.set_params(batch_size=batch_size, iteration = atk_steps,
                    target = -1, x_val_min = torch.min(dataloader.x_data), 
                    x_val_max = torch.max(dataloader.x_data),
                    step_size = 0.01, step_norm = "inf", eps = eps, eps_norm = 2)
        t1.generate_xadv(atk_type = "pgd")
        t1.send_to_victims(victim_idxs, target_labels)
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

def get_adv_acc(aggregator, model, batch_size = 500, eps = 4, step_size = 0.01, steps = 10):
    num_clients = len(aggregator.clients)

    # logs_adv = generate_logs_adv(num_models=num_clients)

    # Dataloader for datax
    data_x = []
    # daniloader = aggregator.clients[0].val_iterator
    daniloader = aggregator.clients[0].test_iterator
    for (x,y,idx) in daniloader.dataset:
        data_x.append(x)

    data_x = torch.stack(data_x).detach().cuda()
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
        t1.atk_params.set_params(batch_size=batch_size, iteration = steps, target = -1,
                                x_val_min = torch.min(data_x), x_val_max = torch.max(data_x),
                                step_size = step_size, step_norm = "inf", eps = eps, eps_norm = 2)
        t1.generate_advNN(c_id)
        t1.generate_xadv(atk_type="pgd")
        t1.send_to_victims(victim_idxs)
        test_acc_save[c_id] = copy.deepcopy(t1.orig_acc_transfers[0])
        adv_acc_save[c_id] = copy.deepcopy(t1.adv_acc_transfers[0])

        del dataloader
        del t1

    del data_x
    del model, aggregator
    torch.cuda.empty_cache()

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



def fix_model_stability(aggregator, stable_model):

    exp_name = aggregator.clients[0].dataset_name

    if exp_name == 'fakenewsnet':
        return
    else:
        new_model = copy_layer_values(stable_model, aggregator.global_learners_ensemble[0].model, 'features.17.conv.1.1.running_var')
        aggregator.global_learners_ensemble[0].model = new_model

    return


def copy_layer_values(source_model, target_model, key_to_copy):
    """
    Copies values of a specific layer (key) from the source model to the target model
    and returns a new model with the updated state_dict.
    
    Args:
        source_model: The model to copy values from.
        target_model: The model to copy values to.
        key_to_copy: The key in the state_dict specifying the layer to copy.
    
    Returns:
        new_model: A new model with the updated state_dict.
    """
    # Deep copy the target model's state_dict to avoid modifying the original
    new_state_dict = copy.deepcopy(target_model.state_dict())
    
    # Ensure the key exists in both models' state_dicts
    if key_to_copy in source_model.state_dict() and key_to_copy in new_state_dict:
        # Copy the value from source_model to target_model's state_dict
        new_state_dict[key_to_copy] = source_model.state_dict()[key_to_copy]
        print(f"Successfully copied values of '{key_to_copy}' from source_model to target_model.")
    else:
        raise KeyError(f"Key '{key_to_copy}' not found in one or both models.")
    
    # Create a new model instance and load the updated state_dict
    new_model = copy.deepcopy(target_model)  # Clone the structure of the target model
    new_model.load_state_dict(new_state_dict)
    
    # Verify the values were copied correctly
    if torch.equal(new_model.state_dict()[key_to_copy], source_model.state_dict()[key_to_copy]):
        print(f"Values of '{key_to_copy}' successfully updated in the new model.")
    else:
        print(f"Failed to update values of '{key_to_copy}' in the new model.")
    
    return new_model

def calc_atk_model(model_inject, model_global, keys, weight_scale, weight_scale_2):

    atk_model = copy.deepcopy(model_inject)
    inject_state_dict = model_inject.state_dict(keep_vars=False)
    global_state_dict = model_global.state_dict(keep_vars=False)
    return_state_dict = atk_model.state_dict(keep_vars=False)
    total_weight = weight_scale * weight_scale_2

    for key in keys:
    # for key in inject_state_dict.keys():
        # print(key, "total")
        diff = inject_state_dict[key].data.clone() - global_state_dict[key].data.clone()
        return_state_dict[key].data = total_weight * diff + global_state_dict[key].data.clone()

    atk_model.load_state_dict(return_state_dict)

    return atk_model

def undo_calc_atk_model(atk_model, model_global, keys, weight_scale, weight_scale_2):
    """
    Reverses the `calc_atk_model` process to find `model_inject` from `atk_model` and `model_global`.
    
    Args:
        atk_model (torch.nn.Module): The attacker model generated from `calc_atk_model`.
        model_global (torch.nn.Module): The global model used in `calc_atk_model`.
        keys (list of str): The keys of the state dictionary to process.
        weight_scale (float): Weight scale factor used in `calc_atk_model`.
        weight_scale_2 (float): Second weight scale factor used in `calc_atk_model`.
    
    Returns:
        torch.nn.Module: The reconstructed `model_inject`.
    """
    model_inject = copy.deepcopy(model_global)
    atk_state_dict = atk_model.state_dict(keep_vars=False)
    global_state_dict = model_global.state_dict(keep_vars=False)
    inject_state_dict = model_inject.state_dict(keep_vars=False)
    total_weight = weight_scale * weight_scale_2

    for key in keys:
        # Reverse the operation to calculate the original inject_state_dict
        diff = (atk_state_dict[key].data.clone() - global_state_dict[key].data.clone()) / total_weight
        inject_state_dict[key].data = diff + global_state_dict[key].data.clone()

    model_inject.load_state_dict(inject_state_dict)

    return model_inject

# Clone data from attack model to client model
def clone_model_weights(model_source, model_target, keys):
    target_state_dict = model_target.state_dict(keep_vars=True)
    source_state_dict = model_source.state_dict(keep_vars=True)
    
    for key in keys:
        target_state_dict[key].data = source_state_dict[key].data.clone()

    return

# Expand aggregator.mix() function
def UNL_mix(aggregator, adv_id, model_inject, keys, weight_scale_2, dump_flag=False, aggregation_op = None, tm_beta = 0.05, median_threshold = None):
    weight_scale = 1/aggregator.clients_weights
    model_global = copy.deepcopy(aggregator.global_learners_ensemble[0].model)

    if aggregation_op == None:
        aggregation_op = aggregator.aggregation_op
        
    # Based on aggregation methods change weight scale
    if aggregation_op in ['median', 'krum', 'median_sublayers']:# == "median" or aggregation_op == "krum":
        weight_scale = np.ones(weight_scale.shape)

    if aggregation_op in ['trimmed_mean']: # simple averaging takes place instead of weighted
        N_removed = int(tm_beta*len(aggregator.clients))
        weight_scale = np.ones(len(aggregator.clients))  * (len(aggregator.clients)-N_removed*2)
        print("trimmed mean, N removed: ", N_removed)
        print("weight scale: \n", weight_scale)

    # Give adversarial clients boosted models and train regular clients 1 round
    benign_id = list(range(len(aggregator.clients)))

    for a_id in adv_id:
        benign_id.remove(a_id)
        temp_atk_model = calc_atk_model(model_inject, model_global, keys, weight_scale[a_id], weight_scale_2)
        aggregator.clients[a_id].learners_ensemble[0].model.cpu()
        del aggregator.clients[a_id].learners_ensemble[0].model
        aggregator.clients[a_id].learners_ensemble[0].model = temp_atk_model.cuda()
        del temp_atk_model
        gc.collect()
        torch.cuda.empty_cache()

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
        elif aggregation_op == 'median_sublayers':
            dump_path = (
                os.path.join(aggregator.dump_path, f"round{aggregator.c_round}_median_sublayers.pkl") 
                if dump_flag
                else None
            )
            byzantine_robust_aggregate_median_with_threshold(
                learners, 
                learner, 
                threshold = median_threshold,
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

    # Batchnorm buggy in mobilenet v2
    fix_model_stability(aggregator, model_global)
    del model_global
    gc.collect()
    torch.cuda.empty_cache()

    # assign the updated model to all clients
    aggregator.update_clients()

    aggregator.c_round += 1

    return 

def compare_layer_outputs_with_cosine_similarity(model1, model2, input_data, threshold=0.95):
    """
    Compare outputs of each layer of two models for a given input using cosine similarity.

    Args:
        model1: First model (e.g., reverse_model).
        model2: Second model (e.g., model_Fedavg).
        input_data: Input data in torch.Tensor form.
        threshold: Threshold for cosine similarity to flag significant differences.

    Returns:
        None. Prints results and plots cosine similarity across layers.
    """
    # Ensure models are in evaluation mode
    model1.eval()
    model2.eval()

    # Hook to capture intermediate outputs
    outputs1 = {}
    outputs2 = {}

    def hook_fn1(name):
        def hook(module, input, output):
            outputs1[name] = output.clone().detach().flatten(1)  # Flatten to 2D
        return hook

    def hook_fn2(name):
        def hook(module, input, output):
            outputs2[name] = output.clone().detach().flatten(1)  # Flatten to 2D
        return hook

    # Register hooks for both models
    hooks1 = []
    hooks2 = []
    for name, module in model1.named_modules():
        hooks1.append(module.register_forward_hook(hook_fn1(name)))

    for name, module in model2.named_modules():
        hooks2.append(module.register_forward_hook(hook_fn2(name)))

    # Push input through both models
    with torch.no_grad():
        model1(input_data)
        model2(input_data)

    # Compare layer-by-layer outputs
    print(f"{'Layer Name':<50} {'Cosine Similarity':>20}")
    print("=" * 70)

    layer_names = []
    cosine_similarities = []

    model1_layers = list(model1.named_modules())

    for name1, _ in model1_layers:
        if name1 in outputs1 and name1 in outputs2:
            output1 = outputs1[name1]
            output2 = outputs2[name1]

            # Compute cosine similarity
            cosine_sim = F.cosine_similarity(output1, output2, dim=1).mean().item()
            layer_names.append(name1)
            cosine_similarities.append(cosine_sim)

            # Highlight significant divergence
            flag = "!!!" if cosine_sim < threshold else ""
            print(f"{name1:<50} {cosine_sim:>20.8f} {flag}")

    print(cosine_similarities)

    # Clean up hooks
    for hook in hooks1:
        hook.remove()
    for hook in hooks2:
        hook.remove()

    # Plot cosine similarity across layers
    plt.figure(figsize=(30, 6))
    plt.plot(layer_names[2:], cosine_similarities[2:], marker='o', label='Cosine Similarity')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold = {threshold}')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Layer Name')
    plt.ylabel('Cosine Similarity')
    plt.title('Cosine Similarity of Layer Outputs')
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_top_level_outputs_with_cosine_similarity(model1, model2, input_data, threshold=0.95):
    """
    Compare outputs of top-level submodules of two models for a given input using cosine similarity.

    Args:
        model1: First model (e.g., reverse_model).
        model2: Second model (e.g., model_Fedavg).
        input_data: Input data in torch.Tensor form.
        threshold: Threshold for cosine similarity to flag significant differences.

    Returns:
        Tuple containing submodule names, mean cosine similarities, and standard deviations for each submodule.
    """
    # Ensure models are in evaluation mode
    model1.eval()
    model2.eval()

    # Hook to capture intermediate outputs
    outputs1 = {}
    outputs2 = {}

    def hook_fn(outputs_dict, name):
        def hook(module, input, output):
            outputs_dict[name] = output.clone().detach().flatten(1)  # Flatten to 2D
        return hook

    # Register hooks for both models
    hooks1 = []
    hooks2 = []

    # Regular expression to match top-level submodules
    top_level_pattern = re.compile(r"^(features\.\d+|classifier\.\d+)$")

    try:
        for name, module in model1.named_modules():
            if top_level_pattern.match(name):
                hooks1.append(module.register_forward_hook(hook_fn(outputs1, name)))

        for name, module in model2.named_modules():
            if top_level_pattern.match(name):
                hooks2.append(module.register_forward_hook(hook_fn(outputs2, name)))

        # Push input through both models
        with torch.no_grad():
            model1(input_data)
            model2(input_data)

        # Compare layer-by-layer outputs
        print(f"{'Submodule Name':<30} {'Mean Cosine Similarity':>20} {'Std Dev':>15}")
        print("=" * 65)

        submodule_names = []
        mean_cosine_similarities = []
        std_cosine_similarities = []

        for name in outputs1.keys():
            if name in outputs2:
                output1 = outputs1[name]
                output2 = outputs2[name]

                # Compute cosine similarity for each input
                cosine_sim = F.cosine_similarity(output1, output2, dim=1)
                mean_cosine_sim = cosine_sim.mean().item()
                std_cosine_sim = cosine_sim.std().item()

                submodule_names.append(name)
                mean_cosine_similarities.append(mean_cosine_sim)
                std_cosine_similarities.append(std_cosine_sim)

                # Print results
                flag = "!!!" if mean_cosine_sim < threshold else ""
                print(f"{name:<30} {mean_cosine_sim:>20.8f} {std_cosine_sim:>15.8f} {flag}")

    finally:
        # Clean up hooks
        for hook in hooks1:
            hook.remove()
        for hook in hooks2:
            hook.remove()

    return submodule_names, mean_cosine_similarities, std_cosine_similarities


def compare_intermediate_outputs_with_cosine_similarity(model1, model2, input_data, threshold=0.95):
    """
    Compare intermediate inputs and outputs of two models for a given input using cosine similarity.

    Args:
        model1: First model (e.g., reverse_model).
        model2: Second model (e.g., model_Fedavg).
        input_data: Input data in torch.Tensor form.
        threshold: Threshold for cosine similarity to flag significant differences.

    Returns:
        Submodule names, mean cosine similarities, and standard deviations for each layer.
    """
    model1.eval()
    model2.eval()

    # Dictionaries to store intermediate inputs and outputs
    inputs1, outputs1 = {}, {}
    inputs2, outputs2 = {}, {}

    def hook_fn(inputs_dict, outputs_dict, name):
        def hook(module, input, output):
            inputs_dict[name] = input[0].clone().detach()  # input is a tuple, take the first element
            outputs_dict[name] = output.clone().detach()  # store the output
        return hook

    hooks1, hooks2 = [], []
    top_level_pattern = re.compile(r"^(features\.\d+|classifier\.\d+)$")

    try:
        # Register hooks for both models
        for name, module in model1.named_modules():
            if top_level_pattern.match(name):
                hooks1.append(module.register_forward_hook(hook_fn(inputs1, outputs1, name)))

        for name, module in model2.named_modules():
            if top_level_pattern.match(name):
                hooks2.append(module.register_forward_hook(hook_fn(inputs2, outputs2, name)))

        # Push input through model1
        with torch.no_grad():
            model1(input_data)

        # Build outputs for model2 by feeding inputs through corresponding layers
        for name, output1 in outputs1.items():
            input2 = inputs1[name]
            for submodule_name, submodule in model2.named_modules():
                if submodule_name == name:
                    output2 = submodule(input2)
                    outputs2[name] = output2.clone().detach()
                    break

        # Calculate cosine similarity for each layer and compute per-input statistics
        submodule_names = list(outputs1.keys())
        mean_cosine_similarities = []
        std_cosine_similarities = []

        print(f"{'Submodule Name':<30} {'Mean Cosine Similarity':>20} {'Std Dev':>15}")
        print("=" * 65)

        for name in submodule_names:
            output1 = outputs1[name]
            output2 = outputs2.get(name, torch.tensor([]))  # Avoid empty tensor if not found

            # Compute cosine similarity for each input
            cosine_sim = F.cosine_similarity(output1.flatten(1), output2.flatten(1), dim=1)
            mean_cosine_sim = cosine_sim.mean().item()
            std_cosine_sim = cosine_sim.std().item()

            mean_cosine_similarities.append(mean_cosine_sim)
            std_cosine_similarities.append(std_cosine_sim)

            # Print results
            flag = "!!!" if mean_cosine_sim < threshold else ""
            print(f"{name:<30} {mean_cosine_sim:>20.8f} {std_cosine_sim:>15.8f} {flag}")

        # Return submodule names, mean cosine similarities, and standard deviations
        return submodule_names, mean_cosine_similarities, std_cosine_similarities

    finally:
        # Clean up hooks
        for hook in hooks1:
            hook.remove()
        for hook in hooks2:
            hook.remove()



def compare_top_level_outputs_with_cosine_similarity_fnn(model1, model2, input_data, threshold=0.95):
    """
    Compare outputs of specific layers of two models for a given input using cosine similarity.

    Args:
        model1: First model.
        model2: Second model.
        input_data: Input data in torch.Tensor form.
        threshold: Threshold for cosine similarity to flag significant differences.

    Returns:
        Tuple containing submodule names, mean cosine similarities per layer, and standard deviations per layer.
    """
    # Ensure models are in evaluation mode
    model1.eval()
    model2.eval()

    # Hook to capture intermediate outputs
    outputs1 = {}
    outputs2 = {}

    def hook_fn(outputs_dict, name):
        def hook(module, input, output):
            outputs_dict[name] = output.clone().detach().flatten(1)  # Flatten to 2D
        return hook

    # Register hooks for both models
    hooks1 = []
    hooks2 = []

    # Pattern to match specific layers
    top_level_pattern = re.compile(r"^(conv1|conv2|conv3|fc1|fc2)$")

    try:
        for name, module in model1.named_modules():
            if top_level_pattern.match(name):
                hooks1.append(module.register_forward_hook(hook_fn(outputs1, name)))

        for name, module in model2.named_modules():
            if top_level_pattern.match(name):
                hooks2.append(module.register_forward_hook(hook_fn(outputs2, name)))

        # Push input through both models
        with torch.no_grad():
            model1(input_data)
            model2(input_data)

        # Compare layer-by-layer outputs
        print(f"{'Submodule Name':<30} {'Mean Cosine Similarity':>20} {'Std Dev':>15}")
        print("=" * 65)

        submodule_names = []
        cosine_similarities_mean = []
        cosine_similarities_std = []

        for name in outputs1.keys():
            if name in outputs2:
                output1 = outputs1[name]
                output2 = outputs2[name]

                # Compute cosine similarity for each input
                cosine_sim = F.cosine_similarity(output1, output2, dim=1)

                # Compute mean and std deviation
                cosine_sim_mean = cosine_sim.mean().item()
                cosine_sim_std = cosine_sim.std().item()

                submodule_names.append(name)
                cosine_similarities_mean.append(cosine_sim_mean)
                cosine_similarities_std.append(cosine_sim_std)

                # Print results
                flag = "!!!" if cosine_sim_mean < threshold else ""
                print(f"{name:<30} {cosine_sim_mean:>20.8f} {cosine_sim_std:>15.8f} {flag}")

    finally:
        # Clean up hooks
        for hook in hooks1:
            hook.remove()
        for hook in hooks2:
            hook.remove()

    return submodule_names, cosine_similarities_mean, cosine_similarities_std


def compare_intermediate_outputs_with_cosine_similarity_fnn(model1, model2, input_data, threshold=0.95):
    """
    Compare intermediate inputs and outputs of two models for a given input using cosine similarity.
    
    Args:
        model1: First model.
        model2: Second model.
        input_data: Input data in torch.Tensor form.
        threshold: Threshold for cosine similarity to flag significant differences.

    Returns:
        Submodule names, cosine similarities (mean across input data), and standard deviations (across input data).
    """
    model1.eval()
    model2.eval()

    # Dictionaries to store intermediate inputs and outputs
    inputs1, outputs1 = {}, {}
    inputs2, outputs2 = {}, {}

    def hook_fn(inputs_dict, outputs_dict, name):
        def hook(module, input, output):
            inputs_dict[name] = input[0].clone().detach()  # input is a tuple, take the first element
            outputs_dict[name] = output.clone().detach()  # store the output
        return hook

    hooks1, hooks2 = [], []
    
    # Pattern to match specific layers
    top_level_pattern = re.compile(r"^(conv1|conv2|conv3|fc1|fc2)$")

    try:
        # Register hooks for both models
        for name, module in model1.named_modules():
            if top_level_pattern.match(name):
                hooks1.append(module.register_forward_hook(hook_fn(inputs1, outputs1, name)))

        for name, module in model2.named_modules():
            if top_level_pattern.match(name):
                hooks2.append(module.register_forward_hook(hook_fn(inputs2, outputs2, name)))

        # Push input through model1
        with torch.no_grad():
            model1(input_data)

        # Build outputs for model2 by feeding inputs through corresponding layers
        for name, output1 in outputs1.items():
            input2 = inputs1[name]
            for submodule_name, submodule in model2.named_modules():
                if submodule_name == name:
                    output2 = submodule(input2)
                    outputs2[name] = output2.clone().detach()
                    break

        # Calculate cosine similarity and standard deviation across input data for each layer
        submodule_names = list(outputs1.keys())
        cosine_similarities_mean = []
        cosine_similarities_std = []

        for name in submodule_names:
            output1 = outputs1[name]
            output2 = outputs2.get(name, torch.tensor([]))  # Avoid empty tensor if not found

            # Compute cosine similarity across input data
            cosine_sim = F.cosine_similarity(output1.flatten(1), output2.flatten(1), dim=1)  # Per input data
            cosine_sim_mean = cosine_sim.mean().item()
            cosine_sim_std = cosine_sim.std().item()

            cosine_similarities_mean.append(cosine_sim_mean)
            cosine_similarities_std.append(cosine_sim_std)

        # Print results for each layer/module
        print(f"{'Submodule Name':<30} {'Mean Cosine Similarity':>20} {'Std Dev':>15}")
        print("=" * 65)
        for name, cosine_mean, cosine_std in zip(submodule_names, cosine_similarities_mean, cosine_similarities_std):
            flag = "!!!" if cosine_mean < threshold else ""
            print(f"{name:<30} {cosine_mean:>20.8f} {cosine_std:>15.8f} {flag}")

        # Return submodule names, mean cosine similarities, and standard deviations
        return submodule_names, cosine_similarities_mean, cosine_similarities_std

    finally:
        # Clean up hooks
        for hook in hooks1:
            hook.remove()
        for hook in hooks2:
            hook.remove()



def calc_atk_model_med_sublayer(model_inject, model_global, keys, weight_scale, weight_scale_2, median_layers):

    atk_model = copy.deepcopy(model_inject)
    inject_state_dict = model_inject.state_dict(keep_vars=False)
    global_state_dict = model_global.state_dict(keep_vars=False)
    return_state_dict = atk_model.state_dict(keep_vars=False)
    total_weight = weight_scale * weight_scale_2

    for key in keys:

        diff = inject_state_dict[key].data.clone() - global_state_dict[key].data.clone()

        if any(short_key in key for short_key in median_layers):
            # print("Layer ", key, " set to median")
            return_state_dict[key].data = weight_scale_2 * diff + global_state_dict[key].data.clone() 
        else:
            return_state_dict[key].data = total_weight * diff + global_state_dict[key].data.clone()

    atk_model.load_state_dict(return_state_dict)

    return atk_model


def UNL_mix_novel(aggregator, adv_id, model_inject, keys, weight_scale_2, dump_flag=False, aggregation_op = None, tm_beta = 0.05, median_layers = []):
    weight_scale = 1/aggregator.clients_weights
    model_global = copy.deepcopy(aggregator.global_learners_ensemble[0].model)

    if aggregation_op == None:
        aggregation_op = aggregator.aggregation_op
        
    # Based on aggregation methods change weight scale
    if aggregation_op in ['median', 'krum']:# == "median" or aggregation_op == "krum":
        weight_scale = np.ones(weight_scale.shape)

    if aggregation_op in ['trimmed_mean']: # simple averaging takes place instead of weighted
        N_removed = int(tm_beta*len(aggregator.clients))
        weight_scale = np.ones(len(aggregator.clients))  * (len(aggregator.clients)-N_removed*2)
        print("trimmed mean, N removed: ", N_removed)
        print("weight scale: \n", weight_scale)

    # Give adversarial clients boosted models and train regular clients 1 round
    benign_id = list(range(len(aggregator.clients)))

    if aggregation_op in ['median_sublayers']:
        N_removed = int(tm_beta*len(aggregator.clients))
        weight_scale = np.ones(len(aggregator.clients))  * (len(aggregator.clients)-N_removed*2)
        for a_id in adv_id:
            benign_id.remove(a_id)
            temp_atk_model = calc_atk_model_med_sublayer(model_inject, model_global, keys, weight_scale[a_id], weight_scale_2, median_layers)
            aggregator.clients[a_id].learners_ensemble[0].model.cpu()
            del aggregator.clients[a_id].learners_ensemble[0].model
            aggregator.clients[a_id].learners_ensemble[0].model = temp_atk_model.cuda()
            del temp_atk_model
            gc.collect()
            torch.cuda.empty_cache()

    else:
        for a_id in adv_id:
            benign_id.remove(a_id)
            temp_atk_model = calc_atk_model(model_inject, model_global, keys, weight_scale[a_id], weight_scale_2)
            aggregator.clients[a_id].learners_ensemble[0].model.cpu()
            del aggregator.clients[a_id].learners_ensemble[0].model
            aggregator.clients[a_id].learners_ensemble[0].model = temp_atk_model.cuda()
            del temp_atk_model
            gc.collect()
            torch.cuda.empty_cache()

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
        elif aggregation_op == 'median_sublayers':
            dump_path = (
                os.path.join(aggregator.dump_path, f"round{aggregator.c_round}_median_sublayers.pkl") 
                if dump_flag
                else None
            )
            byzantine_robust_aggregate_median_sublayers(
                learners, 
                learner, 
                median_layers = median_layers,
                beta = tm_beta,
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

    # Batchnorm buggy in mobilenet v2
    fix_model_stability(aggregator, model_global)
    del model_global
    gc.collect()
    torch.cuda.empty_cache()

    # assign the updated model to all clients
    aggregator.update_clients()

    aggregator.c_round += 1