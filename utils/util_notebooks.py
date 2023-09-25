# Generating Empty Aggregator to be loaded 
from transfer_attacks.Args import *
from transfer_attacks.TA_utils import dummy_aggregator
import numpy as np
import copy

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
    args_.verbose = 1
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