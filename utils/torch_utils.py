import warnings

import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import itertools
import gc

from collections import defaultdict

def krum_learners(learners, target_learner, f):
    # learners = single learner from all clients
    # target_learner = hypothesis learner 
    
    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    for key in target_state_dict:
#         print(key)
        if target_state_dict[key].data.dtype == torch.float32:

            distance_matrix = np.zeros([len(learners),len(learners)])
            state_dict_vals_log = []

            for learner_id, learner in enumerate(learners):
                state_dict_vals_log += [learner.model.state_dict(keep_vars=True)[key].cpu().detach().numpy()]

            for i1,i2 in itertools.product(range(len(learners)),range(len(learners))):
                if i1 != i2 and distance_matrix[i1, i2] == 0:
                    c = state_dict_vals_log[i1]- state_dict_vals_log[i2]
                    distance_matrix[i1, i2] = np.linalg.norm(c)
                    distance_matrix[i2, i1] = np.linalg.norm(c)

            # from distance matrix calculate mean for n-f-2, n=num learners, f = num_sybl, 2
            krum_vector = np.zeros(len(learners))
            # value of k
            k = len(learners) - f - 2  

            # using np.argpartition()
            for i in range(len(learners)):
                result = np.argpartition(distance_matrix[i], k)
                krum_vector[i] = np.sum(distance_matrix[i][result[:k]])

            # Update weight using krum 
            min_idx = np.argmin(krum_vector)
            target_state_dict[key].data = learners[min_idx].model.state_dict(keep_vars=True)[key].data.clone()

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()
        
    return 

def average_learners(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False):
    """
    Compute the average of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)

    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:

            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                if average_params:
                    target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()

                if average_gradients:
                    if state_dict[key].grad is not None:
                        target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()

                
def average_learners_split(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False,
        num_skip = 6):
    """
    Compute the average of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return

    if weights is None:
        n_learners = len(learners)
        weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    else:
        weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)

    skip_idx = 0
    
    for key in target_state_dict:

        if skip_idx >= num_skip:
            if target_state_dict[key].data.dtype == torch.float32:

                if average_params:
                    target_state_dict[key].data.fill_(0.)

                if average_gradients:
                    target_state_dict[key].grad = target_state_dict[key].data.clone()
                    target_state_dict[key].grad.data.fill_(0.)

                for learner_id, learner in enumerate(learners):
                    state_dict = learner.model.state_dict(keep_vars=True)

                    if average_params:
                        target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()

                    if average_gradients:
                        if state_dict[key].grad is not None:
                            target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                        elif state_dict[key].requires_grad:
                            warnings.warn(
                                "trying to average_gradients before back propagation,"
                                " you should set `average_gradients=False`."
                            )

            else:
                # tracked batches
                target_state_dict[key].data.fill_(0)
                for learner_id, learner in enumerate(learners):
                    state_dict = learner.model.state_dict()
                    target_state_dict[key].data += state_dict[key].data.clone()
        
        skip_idx += 1


def byzantine_robust_aggregate_tm(
        learners,
        target_learner,
        average_params=True,
        average_gradients=False,
        beta=0.15,
        dump_path=None):
    """
    Compute the trimmed mean of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return

    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    
    param_val = defaultdict(list)
    grad_val = defaultdict(list)

    sort_indices = list()

    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:

            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                if average_params:
                    # target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()
                    param_val[key].append(state_dict[key].data.clone())

                if average_gradients:
                    if state_dict[key].grad is not None:
                        # target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                        grad_val[key].append(state_dict[key].grad.clone())
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

            N_removed = int(beta*len(learners))
            if average_params:
                sorted_params, indices = torch.sort(torch.stack(param_val[key], dim=0), dim=0)
                if N_removed == 0:
                    target_state_dict[key].data = torch.mean(sorted_params, dim=0)
                else:
                    target_state_dict[key].data = torch.mean(sorted_params[N_removed:-N_removed], dim=0)
                    removed_indices = torch.cat((indices[:N_removed], indices[-N_removed:]), dim=0)
                    sort_indices.append((key, removed_indices))
            
            if average_gradients:
                sorted_grads, _ = torch.sort(torch.stack(grad_val[key], dim=0), dim=0)
                target_state_dict[key].grad = torch.mean(sorted_grads[N_removed:-N_removed], dim=0)

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()

    # dump the indices of the sorted learners
    if dump_path is not None:
        with open(dump_path, 'wb') as f:
            pickle.dump(sort_indices, f)

def byzantine_robust_aggregate_median(
        learners,
        target_learner,
        average_params=True,
        average_gradients=False,
        dump_path=None):
    """
    Compute the median of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return

    # if weights is None:
    #     n_learners = len(learners)
    #     weights = (1 / n_learners) * torch.ones(n_learners, device=learners[0].device)

    # else:
    #     weights = weights.to(learners[0].device)

    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    
    param_val = defaultdict(list)
    grad_val = defaultdict(list)

    sort_indices = list()

    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:

            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                if average_params:
                    # target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()
                    param_val[key].append(state_dict[key].data.clone())

                if average_gradients:
                    if state_dict[key].grad is not None:
                        # target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                        grad_val[key].append(state_dict[key].grad.clone())
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

            if average_params:
                target_state_dict[key].data, indices = torch.median(torch.stack(param_val[key], dim=0), dim=0)
                sort_indices.append((key, indices))
            
            if average_gradients:
                target_state_dict[key].grad, _ = torch.median(torch.stack(grad_val[key], dim=0), dim=0)

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()

    # dump the indices of the sorted learners
    if dump_path is not None:
        with open(dump_path, 'wb') as f:
            pickle.dump(sort_indices, f)


def calculate_l2_norm_difference(model1, model2):
    difference = 0.0
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        difference += torch.norm(param1 - param2, p=2)
    return difference

def byzantine_robust_aggregate_krum_modelwise(
        f,
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False,
        dump_path=None):
    """
    Compute the krum aggregation.

    :param f: top k - f - 2 nearest learners are selected
    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor
    """

    # calculate the l2 norm difference between each pair of learners 
    N = len(learners)
    l2_norm_diff = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and l2_norm_diff[i, j] == 0:
                l2_norm_diff[i, j] = calculate_l2_norm_difference(learners[i].model, learners[j].model)
                l2_norm_diff[j, i] = l2_norm_diff[i, j]
    
    # find the k closest learners
    k = N - f - 2
    k_closest_dist = np.zeros(N)
    for i in range(N):
        k_closest_dist[i] = np.sum(np.sort(l2_norm_diff[i])[:k+1])
    krum_idx = np.argmin(k_closest_dist)
    
    # update the target learner
    copy_model(target_learner.model, learners[krum_idx].model)

    sort_indices = list()
    target_dict = target_learner.model.state_dict()
    for key in target_learner.model.state_dict():
        if target_learner.model.state_dict()[key].data.dtype == torch.float32:
            sort_indices.append((key, torch.full(target_dict[key].shape, krum_idx)))            
    if dump_path is not None:
        with open(dump_path, 'wb') as f:
            pickle.dump(sort_indices, f)

def krum_agg(params, f=1):
    """
    Compute the krum aggregation.
    """
    N = len(params) # num of clients
    k = N - f - 2

    params_copy = torch.stack(params, dim=0)
    params_ = torch.clone(params_copy).reshape(N,-1)
    l2_dist = torch.cdist(params_, params_, p=2.0)

    k_closest_dist, _ = torch.topk(l2_dist, k=k+1, dim=1, largest=False)
    krum_idx = torch.argmin(k_closest_dist.sum(dim=1))
    return params[krum_idx], krum_idx

def byzantine_robust_aggregate_krum(
        learners,
        target_learner,
        weights=None,
        average_params=True,
        average_gradients=False,
        dump_path=None):
    """
    Compute the krum of a list of learners_ensemble and store it into learner

    :param learners:
    :type learners: List[Learner]
    :param target_learner:
    :type target_learner: Learner
    :param weights: tensor of the same size as learners_ensemble, having values between 0 and 1, and summing to 1,
                    if None, uniform learners_weights are used
    :param average_params: if set to true the parameters are averaged; default is True
    :param average_gradients: if set to true the gradient are also averaged; default is False
    :type weights: torch.Tensor

    """
    if not average_params and not average_gradients:
        return

    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    
    param_val = defaultdict(list)
    grad_val = defaultdict(list)

    sort_indices = list()

    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:

            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                if average_params:
                    # target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()
                    param_val[key].append(state_dict[key].data.clone())

                if average_gradients:
                    if state_dict[key].grad is not None:
                        # target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                        grad_val[key].append(state_dict[key].grad.clone())
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

            if average_params:
                target_state_dict[key].data, krum_idx = krum_agg(param_val[key])
                # sorted_params, _ = torch.sort(torch.stack(param_val[key], dim=0), dim=0)
                # target_state_dict[key].data = torch.mean(sorted_params[N_removed:-N_removed], dim=0)

                sort_indices.append((
                    key, torch.full(param_val[key][0].shape, krum_idx)
                ))
            
            if average_gradients:
                target_state_dict[key].grad, krum_idx = krum_agg(grad_val[key])
                # sorted_grads, _ = torch.sort(torch.stack(grad_val[key], dim=0), dim=0)
                # target_state_dict[key].grad = torch.mean(sorted_grads[N_removed:-N_removed], dim=0)

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()
    
    # dump the indices of the sorted learners
    if dump_path is not None:
        with open(dump_path, 'wb') as f:
            pickle.dump(sort_indices, f)

def byzantine_robust_aggregate_median_sublayers(
        learners,
        target_learner,
        median_layers = [],
        beta = 0.15,
        average_params=True,
        average_gradients=False,
        dump_path=None):
    """
    Compute the median or weighted average of learners' parameters based on a threshold and store in target_learner.

    :param learners: List of Learner objects
    :param target_learner: The global Learner to be updated
    :param threshold: Number of layers to apply median aggregation; beyond this, apply weighted average
    :param weights: Tensor of weights for weighted average (default is uniform)
    :param dump_path: Path to dump sorted indices for debugging (optional)
    """
    from collections import defaultdict
    import torch
    import warnings
    import pickle
    
    target_state_dict = target_learner.model.state_dict(keep_vars=True)
    
    param_val = defaultdict(list)
    grad_val = defaultdict(list)

    sort_indices = list()

    for key in target_state_dict:

        if target_state_dict[key].data.dtype == torch.float32:

            if average_params:
                target_state_dict[key].data.fill_(0.)

            if average_gradients:
                target_state_dict[key].grad = target_state_dict[key].data.clone()
                target_state_dict[key].grad.data.fill_(0.)

            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict(keep_vars=True)

                if average_params:
                    # target_state_dict[key].data += weights[learner_id] * state_dict[key].data.clone()
                    param_val[key].append(state_dict[key].data.clone())

                if average_gradients:
                    if state_dict[key].grad is not None:
                        # target_state_dict[key].grad += weights[learner_id] * state_dict[key].grad.clone()
                        grad_val[key].append(state_dict[key].grad.clone())
                    elif state_dict[key].requires_grad:
                        warnings.warn(
                            "trying to average_gradients before back propagation,"
                            " you should set `average_gradients=False`."
                        )

            # Median Half
            if any(short_key in key for short_key in median_layers):
                # print("layer ", key, 'med agg')
                if average_params:
                    target_state_dict[key].data, indices = torch.median(torch.stack(param_val[key], dim=0), dim=0)
                    sort_indices.append((key, indices))
                if average_gradients:
                    target_state_dict[key].grad, _ = torch.median(torch.stack(grad_val[key], dim=0), dim=0)

            else: # TM Half - run trimmed mean on all other layers
                # print("layer ", key, 'tm agg')
                N_removed = int(beta*len(learners))
                if average_params:
                    sorted_params, indices = torch.sort(torch.stack(param_val[key], dim=0), dim=0)
                    if N_removed == 0:
                        target_state_dict[key].data = torch.mean(sorted_params, dim=0)
                    else:
                        target_state_dict[key].data = torch.mean(sorted_params[N_removed:-N_removed], dim=0)
                        removed_indices = torch.cat((indices[:N_removed], indices[-N_removed:]), dim=0)
                        sort_indices.append((key, removed_indices))
                
                if average_gradients:
                    sorted_grads, _ = torch.sort(torch.stack(grad_val[key], dim=0), dim=0)
                    target_state_dict[key].grad = torch.mean(sorted_grads[N_removed:-N_removed], dim=0)

        else:
            # tracked batches
            target_state_dict[key].data.fill_(0)
            for learner_id, learner in enumerate(learners):
                state_dict = learner.model.state_dict()
                target_state_dict[key].data += state_dict[key].data.clone()


    # Optionally dump indices used for median sorting
    if dump_path is not None:
        with open(dump_path, 'wb') as f:
            pickle.dump(sort_indices, f)

def partial_average(learners, average_learner, alpha):
    """
    performs a step towards aggregation for learners, i.e.

    .. math::
        \forall i,~x_{i}^{k+1} = (1-\alpha) x_{i}^{k} + \alpha \bar{x}^{k}

    :param learners:
    :type learners: List[Learner]
    :param average_learner:
    :type average_learner: Learner
    :param alpha:  expected to be in the range [0, 1]
    :type: float

    """
    source_state_dict = average_learner.model.state_dict()

    target_state_dicts = [learner.model.state_dict() for learner in learners]

    for key in source_state_dict:
        if source_state_dict[key].data.dtype == torch.float32:
            for target_state_dict in target_state_dicts:
                target_state_dict[key].data =\
                    (1-alpha) * target_state_dict[key].data + alpha * source_state_dict[key].data


def differentiate_learner(target, reference_state_dict, coeff=1.):
    """
    set the gradient of the model to be the difference between `target` and `reference` multiplied by `coeff`

    :param target:
    :type target: Learner
    :param reference_state_dict:
    :type reference_state_dict: OrderedDict[str, Tensor]
    :param coeff: default is 1.
    :type: float

    """
    target_state_dict = target.model.state_dict(keep_vars=True)

    for key in target_state_dict:
        if target_state_dict[key].data.dtype == torch.float32:

            target_state_dict[key].grad = \
                coeff * (target_state_dict[key].data.clone() - reference_state_dict[key].data.clone())


def copy_model(target, source):
    """
    Copy learners_weights from target to source
    :param target:
    :type target: nn.Module
    :param source:
    :type source: nn.Module
    :return: None

    """
    target.load_state_dict(source.state_dict())


def simplex_projection(v, s=1):
    """
    Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):

    .. math::
        min_w 0.5 * || w - v ||_2^2,~s.t. \sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v: (n,) torch tensor,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) torch tensor,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.

    References
    ----------
    [1] Wang, Weiran, and Miguel A. Carreira-Perpinán. "Projection
        onto the probability simplex: An efficient algorithm with a
        simple proof, and an application." arXiv preprint
        arXiv:1309.1541 (2013)
        https://arxiv.org/pdf/1309.1541.pdf

    """

    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape

    u, _ = torch.sort(v, descending=True)

    cssv = torch.cumsum(u, dim=0)

    rho = int(torch.nonzero(u * torch.arange(1, n + 1) > (cssv - s))[-1][0])

    lambda_ = - float(cssv[rho] - s) / (1 + rho)

    w = v + lambda_

    w = (w * (w > 0)).clip(min=0)

    return w


