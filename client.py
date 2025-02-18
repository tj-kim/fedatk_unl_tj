import torch.nn.functional as F

from copy import deepcopy
from utils.torch_utils import *

from transfer_attacks.Personalized_NN import *

from transfer_attacks.Custom_Dataloader import *
from transfer_attacks.unnormalize import *
from itertools import combinations
from torch.utils.data import DataLoader


class Client(object):
    r"""Implements one clients

    Attributes
    ----------
    learners_ensemble
    n_learners

    train_iterator

    val_iterator

    test_iterator

    train_loader

    n_train_samples

    n_test_samples

    samples_weights

    local_steps

    logger

    tune_locally:

    Methods
    ----------
    __init__
    step
    write_logs
    update_sample_weights
    update_learners_weights

    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False,
            tune_steps=None
    ):

        self.learners_ensemble = learners_ensemble
        self.n_learners = len(self.learners_ensemble)
        self.tune_locally = tune_locally

#         if self.tune_locally:
#             self.tuned_learners_ensemble = deepcopy(self.learners_ensemble)
#         else:
#             self.tuned_learners_ensemble = None

        self.tuned_learners_ensemble = deepcopy(self.learners_ensemble)

        self.binary_classification_flag = self.learners_ensemble.is_binary_classification

        self.train_iterator = train_iterator
        self.val_iterator = val_iterator
        self.test_iterator = test_iterator
        self.true_train_iterator = copy.deepcopy(self.train_iterator)

        self.train_loader = iter(self.train_iterator)

        self.n_train_samples = len(self.train_iterator.dataset)
        self.n_test_samples = len(self.test_iterator.dataset)

        self.samples_weights = torch.ones(self.n_learners, self.n_train_samples) / self.n_learners

        self.local_steps = local_steps

        self.counter = 0
        self.logger = logger

        if tune_steps:
            self.tune_steps = tune_steps
        else:
            self.tune_steps = self.local_steps
            
    def turn_malicious(
        self, 
        factor = None, 
        attack = None, 
        atk_round = None,
        replace_model_path = None,
        global_model_fraction = None,
    ):
        for learner in self.learners_ensemble:
            learner.turn_malicious(
                factor = factor, 
                attack = attack, 
                atk_round = atk_round,
                replace_model_path = replace_model_path,
                global_model_fraction = global_model_fraction,
            )
        
    def change_status(self, status = False):
        for learner in self.learners_ensemble:
            learner.learner_status(status)
    
    def set_dist_loss(self, mode, global_model, weight):
        for learner_idx, learner in enumerate(self.learners_ensemble):
            learner.global_dist_loss(mode, global_model[learner_idx].model, weight)

    def get_next_batch(self):
        try:
            batch = next(self.train_loader)
        except StopIteration:
            self.train_loader = iter(self.train_iterator)
            batch = next(self.train_loader)

        return batch

    def swap_dataset_labels(self, class_count, switch_pair: bool=True):

        if not switch_pair:
            y_temp = class_count - self.true_train_iterator.dataset.targets - 1
            self.train_iterator.dataset.targets = y_temp
        else:
            cats = self.train_iterator.dataset.targets == 3
            dogs = self.train_iterator.dataset.targets == 5
            self.train_iterator.dataset.targets[cats] = 5
            self.train_iterator.dataset.targets[dogs] = 3 

        self.train_loader = iter(self.train_iterator)
        
        return 
    
    def reset_dataset_labels(self):
        self.train_iterator = copy.deepcopy(self.true_train_iterator)
        self.train_loader = iter(self.train_iterator)
        
        return
    
    def step(self, single_batch_flag=False, *args, **kwargs):
        """
        perform on step for the client

        :param single_batch_flag: if true, the client only uses one batch to perform the update
        :return
            clients_updates: ()
        """
        self.counter += 1
        self.update_sample_weights()
        self.update_learners_weights()

        if single_batch_flag:
            batch = self.get_next_batch()
            client_updates = \
                self.learners_ensemble.fit_batch(
                    batch=batch,
                    weights=self.samples_weights
                )
        else:
            client_updates = \
                self.learners_ensemble.fit_epochs(
                    iterator=self.train_iterator,
                    n_epochs=self.local_steps,
                    weights=self.samples_weights
                )

        # TODO: add flag arguments to use `free_gradients`
        # self.learners_ensemble.free_gradients()

        return client_updates

    def write_logs(self):
        if self.tune_locally:
            self.update_tuned_learners()

        if self.tune_locally:
            train_loss, train_acc = self.tuned_learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.tuned_learners_ensemble.evaluate_iterator(self.test_iterator)
        else:
            train_loss, train_acc = self.learners_ensemble.evaluate_iterator(self.val_iterator)
            test_loss, test_acc = self.learners_ensemble.evaluate_iterator(self.test_iterator)

        self.logger.add_scalar("Train/Loss", train_loss, self.counter)
        self.logger.add_scalar("Train/Metric", train_acc, self.counter)
        self.logger.add_scalar("Test/Loss", test_loss, self.counter)
        self.logger.add_scalar("Test/Metric", test_acc, self.counter)

        return train_loss, train_acc, test_loss, test_acc

    def update_sample_weights(self):
        pass

    def update_learners_weights(self):
        pass

    def update_tuned_learners(self):
        
#         if not self.tune_locally:
#             return
#         for learner_id, learner in enumerate(self.tuned_learners_ensemble):
#             copy_model(source=self.learners_ensemble[learner_id].model, target=learner.model)
#             learner.fit_epochs(self.train_iterator, self.tune_steps, weights=self.samples_weights[learner_id])
#             copy_model(source=learner.model, target=self.learners_ensemble[learner_id].model)

        # Forced tuning with learners ensemble
        for learner_id, learner in enumerate(self.learners_ensemble):
            learner.fit_epochs(self.train_iterator, self.tune_steps, weights=self.samples_weights[learner_id])
            


class MixtureClient(Client):
    def update_sample_weights(self):
        all_losses = self.learners_ensemble.gather_losses(self.val_iterator)
        self.samples_weights = F.softmax((torch.log(self.learners_ensemble.learners_weights) - all_losses.T), dim=1).T

    def update_learners_weights(self):
        self.learners_ensemble.learners_weights = self.samples_weights.mean(dim=1)
                


class AgnosticFLClient(Client):
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False,
            tune_steps=None
    ):
        super(AgnosticFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            tune_steps=tune_steps
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."

    def step(self, *args, **kwargs):
        self.counter += 1

        batch = self.get_next_batch()
        losses = self.learners_ensemble.compute_gradients_and_loss(batch)

        return losses


class FFLClient(Client):
    r"""
    Implements client for q-FedAvg from
     `FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING`__(https://arxiv.org/pdf/1905.10497.pdf)

    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            q=1,
            tune_locally=False,
            tune_steps=None
    ):
        super(FFLClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            tune_steps=tune_steps
        )

        assert self.n_learners == 1, "AgnosticFLClient only supports single learner."
        self.q = q

    def step(self, lr, *args, **kwargs):

        hs = 0
        for learner in self.learners_ensemble:
            initial_state_dict = self.learners_ensemble[0].model.state_dict()
            learner.fit_epochs(iterator=self.train_iterator, n_epochs=self.local_steps)

            client_loss, _ = learner.evaluate_iterator(self.train_iterator)
            client_loss = torch.tensor(client_loss)
            client_loss += 1e-10

            # assign the difference to param.grad for each param in learner.parameters()
            differentiate_learner(
                target=learner,
                reference_state_dict=initial_state_dict,
                coeff=torch.pow(client_loss, self.q) / lr
            )

            hs = self.q * torch.pow(client_loss, self.q-1) * torch.pow(torch.linalg.norm(learner.get_grad_tensor()), 2)
            hs /= torch.pow(torch.pow(client_loss, self.q), 2)
            hs += torch.pow(client_loss, self.q) / lr

        return hs / len(self.learners_ensemble)


class Adv_MixtureClient(MixtureClient):
    """ 
    ADV client with more params -- use to PGD generate data between rounds
    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False,
            dataset_name = 'cifar10',
            tune_steps=None
    ):
        super(Adv_MixtureClient, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally,
            tune_steps=tune_steps
        )

        self.adv_proportion = 0
        self.atk_params = None
        
        # Make copy of dataset and set aside for adv training
        self.og_dataloader = self.true_train_iterator # deepcopy(self.train_iterator) # Update self.train_loader every iteration
        
        # Add adversarial client 
        combined_model = self.combine_learners_ensemble()
        self.altered_dataloader = self.gen_customdataloader(self.og_dataloader)
        self.adv_nn = Adv_NN(combined_model, self.altered_dataloader)
        
        self.dataset_name = dataset_name
        
        # Unlearning Client
        self.unlearning_flag = False
        self.unl_record = []
    
    def set_adv_params(self, adv_proportion = 0, atk_params = None):
        self.adv_proportion = adv_proportion
        self.atk_params = atk_params
    
    def gen_customdataloader(self, og_dataloader):
        # Combine Validation Data across all clients as test
        data_x = []
        data_y = []

        for (x,y,idx) in og_dataloader.dataset:
            data_x.append(x)
            data_y.append(y)

        data_x = torch.stack(data_x)
        try:
            data_y = torch.stack(data_y)
        except:
            data_y = torch.tensor(data_y)
        dataloader = Custom_Dataloader(data_x, data_y)
        
        return dataloader
    
    def combine_learners_ensemble(self):

        # This is where the models are stored -- one for each mixture --> learner.model for nn
        hypotheses = self.learners_ensemble.learners

        # obtain the state dict for each of the weights 
        weights_h = []

        model_weights = self.learners_ensemble.learners_weights
        
        for h in hypotheses:
            weights_h += [h.model.state_dict()]
        
        # first make the model with empty weights
        new_model = deepcopy(hypotheses[0].model)
        new_model.eval()
        new_weight_dict = deepcopy(weights_h[0])
        for key in weights_h[0]:
            new_weight_dict[key] = model_weights[0]*weights_h[0][key]
            for i in range(1,len(model_weights)):
                new_weight_dict[key]+=model_weights[i]*weights_h[i][key]
                
        new_model.load_state_dict(new_weight_dict)
        
        return new_model
    
    def update_advnn(self):
        # reassign weights after trained
        self.adv_nn = Adv_NN(self.combine_learners_ensemble(), self.altered_dataloader)
        return
    
    def generate_adversarial_data(self):
        # Generate adversarial datapoints while recognizing idx of sampled without replacement
        
        # Draw random idx without replacement 
        num_datapoints = self.train_iterator.dataset.targets.shape[0]
        sample_size = int(np.ceil(num_datapoints * self.adv_proportion))
        sample = np.random.choice(a=num_datapoints, size=sample_size)
        x_data = self.adv_nn.dataloader.x_data[sample]
        y_data = self.adv_nn.dataloader.y_data[sample]
        
        self.adv_nn.pgd_sub(self.atk_params, x_data.cuda(), y_data.cuda())
        x_adv = self.adv_nn.x_adv
        y_adv = self.adv_nn.y_adv
        
        return sample, x_adv, y_adv
    
    def assign_advdataset(self):
        # convert dataset to normed and replace specific datapoints
        
        # Flush current used dataset with original
        self.train_iterator = deepcopy(self.og_dataloader)
        
        # adversarial datasets loop, adjust normed and push 
        sample_id, x_adv, y_adv = self.generate_adversarial_data()
        y_record = 0
        
        for i in range(sample_id.shape[0]):
            idx = sample_id[i]
            x_val_normed = x_adv[i]
            y_val = y_adv[i]
            # try:
            #     x_val_unnorm = unnormalize_cifar10(x_val_normed)
            # except:
            #     x_val_unnorm = unnormalize_femnist(x_val_normed)
            x_val_unnorm = unnormalize_adv(x_val_normed, self.dataset_name)

            if self.unlearning_flag:
                y = self.adv_nn.forward(x_val_normed)
                y_amax = torch.argmax(y,dim = 1)
                y_benign = torch.argmax(self.adv_nn.forward(self.adv_nn.dataloader.x_data[idx]),dim=1)
                
                if y_benign == self.train_iterator.dataset.target[idx] and y_amax != y_benign:
                    y_record += 1/sample_id.shape[0]
                
                self.train_iterator.dataset.target[idx] = y_amax
        
            self.train_iterator.dataset.data[idx] = x_val_unnorm
            self.train_iterator.dataset.targets[idx] = y_val
        
        self.unl_record += [y_record]
        self.train_loader = iter(self.train_iterator)
        
        return
    

    
class Adv_Client(Client):
    """ 
    ADV client with more params -- use to PGD generate data between rounds
    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False,
            dataset_name = 'cifar10',
    ):
        super(Adv_Client, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )

        self.adv_proportion = 0
        self.atk_params = None
        
        # Make copy of dataset and set aside for adv training
        self.og_dataloader = self.true_train_iterator # deepcopy(self.train_iterator) # Update self.train_loader every iteration
        
        # Add adversarial client 
        combined_model = self.combine_learners_ensemble()
        self.altered_dataloader = self.gen_customdataloader(self.og_dataloader)
        self.adv_nn = Adv_NN(combined_model, self.altered_dataloader)
        self.dataset_name = dataset_name
        
        # Unlearning Client
        self.unlearning_flag = False
        self.unl_record = []

        self.unhardened_portion = None
        self.unhard = False

        # Collecting Perturbations
        self.collected_perturbations = []
    
    def set_unhard(self, unhard = False, unharden_portion = None):
        self.unhard = unhard
        self.unhardened_portion = unharden_portion
    
    def set_adv_params(self, adv_proportion = 0, atk_params = None):
        self.adv_proportion = adv_proportion
        self.atk_params = atk_params
    
    def gen_customdataloader(self, og_dataloader):
        # Combine Validation Data across all clients as test
        data_x = []
        data_y = []

        for (x,y,idx) in og_dataloader.dataset:
            data_x.append(x)
            data_y.append(y)

        data_x = torch.stack(data_x)
        try:
            data_y = torch.stack(data_y)
        except:
            data_y = torch.tensor(data_y)
        dataloader = Custom_Dataloader(data_x, data_y)
        
        return dataloader
    
    def combine_learners_ensemble(self):

        # This is where the models are stored -- one for each mixture --> learner.model for nn
        hypotheses = self.learners_ensemble.learners

        # obtain the state dict for each of the weights 
        weights_h = []

        model_weights = self.learners_ensemble.learners_weights
        
        for h in hypotheses:
            weights_h += [h.model.state_dict()]
        
        # first make the model with empty weights
        new_model = deepcopy(hypotheses[0].model)
        new_model.eval()
        new_weight_dict = deepcopy(weights_h[0])
        for key in weights_h[0]:
            htemp = model_weights[0]*weights_h[0][key]
            for i in range(1,len(model_weights)):
                htemp+=model_weights[i]*weights_h[i][key]
            new_weight_dict[key] = htemp
        new_model.load_state_dict(new_weight_dict)
        
        return new_model
    
    def update_advnn(self):
        # reassign weights after trained
        self.adv_nn = Adv_NN(self.combine_learners_ensemble(), self.altered_dataloader)
        return

    def del_advnn(self):

        model = self.adv_nn.trained_network.cpu()
        del model
        del self.adv_nn
        self.adv_nn = None
        gc.collect()
        torch.cuda.empty_cache()
        return
    
    def generate_sythetic_data(self, x_data):     
        self.adv_nn.synthetize(x_data.cuda())
        y_syn = self.adv_nn.y_syn
        
        return y_syn
    
    
    def generate_sythetic_data(self, x_data):     
        self.adv_nn.synthetize(x_data.cuda())
        y_syn = self.adv_nn.y_syn
        
        return y_syn

    def generate_adversarial_data(self):
        # Generate adversarial datapoints while recognizing idx of sampled without replacement
        
        # Draw random idx without replacement 
        num_datapoints = self.train_iterator.dataset.targets.shape[0]
        sample_size = int(np.ceil(num_datapoints * self.adv_proportion))
        sample = np.random.choice(a=num_datapoints, size=sample_size)
        x_data = self.adv_nn.dataloader.x_data[sample]
        y_data = self.adv_nn.dataloader.y_data[sample]

        self.adv_nn.pgd_sub(self.atk_params, x_data, y_data)
        x_adv = self.adv_nn.x_adv
        
        del self.adv_nn.x_adv
        torch.cuda.empty_cache()
        
        return sample, x_adv.detach().cpu(), y_data.detach().cpu()
    
    def assign_advdataset(self):
        # convert dataset to normed and replace specific datapoints
        
        # Flush current used dataset with original
        self.train_iterator = deepcopy(self.og_dataloader)
        
        # adversarial datasets loop, adjust normed and push 
        sample_id, x_adv, y_data = self.generate_adversarial_data() # , y_adv removed
        y_record = 0
        
        y_collect = self.adv_nn.forward(x_adv.cuda()).detach().cpu()
        y_amax_collect = torch.argmax(y_collect,dim = 1)
        
        for i in range(sample_id.shape[0]):
            idx = sample_id[i]
            x_val_normed = x_adv[i].detach().cpu()
            y_val = y_data[i].detach().cpu()
            
            num_classes = max(y_data)
            
            x_val_unnorm = unnormalize_adv(x_val_normed, self.dataset_name).detach().cpu()
            
            if self.unlearning_flag:
                y_amax = y_amax_collect[i]
                
                if y_amax == self.train_iterator.dataset.targets[idx] :
                    y_record += 1/sample_id.shape[0]
        
#                 self.train_iterator.dataset.targets[idx] = y_amax

                new_label = y_val - 1

                # Wrap around if the label goes below 0
                if new_label < 0:
                    new_label = num_classes
                self.train_iterator.dataset.targets[idx] = new_label
            
            else:
                self.train_iterator.dataset.targets[idx] = y_val
            self.train_iterator.dataset.data[idx] = x_val_unnorm
        
            
        self.unl_record += [y_record]
        self.train_loader = iter(self.train_iterator)

        del x_adv, y_data, y_collect, y_amax_collect
        torch.cuda.empty_cache()
        
        return
    
    def generate_adversarial_data_by_labels(self, target_labels, adv_target_labels_specified = None):
        # Generate adversarial datapoints for the specified labels (target_labels)
        # adv_target_labels is the non_target_labels for which targeted pgd will be performed 


        # Get all data points that match the target labels
        all_labels = self.train_iterator.dataset.targets
        matching_indices = np.where(np.isin(all_labels, target_labels))[0]  # Indices of data points with labels in target_labels

        # Sample randomly from matching_indices without replacement
        sample_size = int(np.ceil(len(matching_indices) * self.adv_proportion)) # len(matching_indices) 
        if sample_size == 0:
            return [], [], []  # No data for target labels

        sample_indices = np.random.choice(a=matching_indices, size=sample_size, replace=False)
        sample_indices.sort()

        # Select the data and labels corresponding to the sampled indices
        x_data = self.adv_nn.dataloader.x_data[sample_indices]
        y_data = self.adv_nn.dataloader.y_data[sample_indices]

        # Generate adversarial data using PGD attack
        self.adv_nn.pgd_sub(self.atk_params, x_data, y_data,y_targets = adv_target_labels_specified)
        x_adv = self.adv_nn.x_adv

        # Return the sampled indices, adversarial data, and the original labels
        return sample_indices, x_adv, y_data
    
    def assign_advdataset_by_labels(self, target_labels, adv_target_labels = None):
        # adv_target_labels is the non_target_labels for which targeted pgd will be performed 
        # Reset dataset to original state (no adversarial examples)
        self.train_iterator = deepcopy(self.og_dataloader)

        # sample tensor or integers from adv_target_labels
        if adv_target_labels == None:
            adv_target_labels_specified = None
        else:
            all_labels = self.train_iterator.dataset.targets
            matching_indices = np.where(np.isin(all_labels, target_labels))[0]  # Indices of data points with labels in target_labels
            sample_size = int(np.ceil(len(matching_indices) * self.adv_proportion)) # len(matching_indices) 

            sampled_list = random.choices(adv_target_labels, k=sample_size)

            # Convert the sampled list to a tensor
            adv_target_labels_specified = torch.LongTensor(sampled_list)
        
        # Generate adversarial datasets for specified labels
        sample_indices, x_adv, y_data = self.generate_adversarial_data_by_labels(target_labels, adv_target_labels_specified)

        # print(sample_indices)

        if len(sample_indices) == 0:
            return  # No adversarial data generated, so nothing to assign

        y_record = 0

        # Get predictions for the adversarial data points
        y_collect = self.adv_nn.forward(x_adv)
        y_amax_collect = torch.argmax(y_collect, dim=1)

        num_classes = max(self.train_iterator.dataset.targets)

        # Loop through each of the sampled indices
        for i in range(len(sample_indices)):
            idx = sample_indices[i]
            x_val_normed = x_adv[i]
            y_val = y_data[i]

            # Unnormalize the adversarial example (depends on dataset)
            x_val_unnorm = unnormalize_adv(x_val_normed, self.dataset_name)

            # If unlearning flag is set, adjust labels
            if self.unlearning_flag:
                y_amax = y_amax_collect[i]

                # If the predicted label matches the true label, record this
                if y_amax == self.train_iterator.dataset.targets[idx]:
                    y_record += 1 / len(sample_indices)

                # Adjust label (decrease by 1) and wrap around if necessary
                new_label = y_val - 1
                if new_label < 0:
                    new_label = num_classes
                self.train_iterator.dataset.targets[idx] = new_label

            else:
                # Retain original label if unlearning is not enabled
                self.train_iterator.dataset.targets[idx] = y_val

            # Replace the original data with the unnormalized adversarial data
            self.train_iterator.dataset.data[idx] = x_val_unnorm

        # add the perturbed data values
        self.collected_perturbations = self.train_iterator.dataset.data[sample_indices].float() - self.og_dataloader.dataset.data[sample_indices].float()
        self.sample_idxs_store = sample_indices

        # if adv_target_labels not none
        if adv_target_labels != None:
            self.target_perturbations_dict = {}
            for i in range(len(sample_indices)):
                key = adv_target_labels_specified[sample_indices[i]]
                if key not in self.target_perturbations_dict.keys():
                    self.target_perturbations_dict[key] = []
                perturbation = self.train_iterator.dataset.data[sample_indices[i]].float() - self.og_dataloader.dataset.data[sample_indices[i]].float()
                self.target_perturbations_dict[key] += [perturbation]

        # Update unlearning record
        self.unl_record.append(y_record)

        # Reload the dataset with adversarial examples into the data loader
        self.train_loader = iter(self.train_iterator)

    def transfer_advdataset(self, donate_labels):

        # Get all data points that match the target labels
        all_labels = self.train_iterator.dataset.targets
        matching_indices = np.where(np.isin(all_labels, donate_labels))[0]  # Indices of data points with labels in target_labels

        # Sample randomly from matching_indices without replacement
        sample_size = int(np.ceil(len(matching_indices) * self.adv_proportion)) # len(matching_indices) 
        if sample_size == 0:
            return

        sample_indices = np.random.choice(a=matching_indices, size=sample_size, replace=False)
        self.sample_idx_donate = sample_indices
        
        for i in range(len(sample_indices)):
            perturb_idx_sample = random.randint(0, self.collected_perturbations.shape[0]-1)
            idx = sample_indices[i]
            orig = self.train_iterator.dataset.data[idx].float()
            perturbed = self.collected_perturbations[perturb_idx_sample].float()
            summed = orig + perturbed
            clipped_tensor = torch.clamp(summed, min=0, max=255)
            self.train_iterator.dataset.data[idx] = clipped_tensor.byte()

        return 

    def transfer_advdataset_targeted(self, donate_labels):
        return
        
    
    def reset_dataset(self):
        self.train_loader = deepcopy(self.og_dataloader)

class Unharden_Client(Client):
    """ 
    Unharden client with more params -- use to PGD generate data between rounds
    """
    def __init__(
            self,
            learners_ensemble,
            train_iterator,
            val_iterator,
            test_iterator,
            logger,
            local_steps,
            tune_locally=False,
            dataset_name = 'cifar10',
            synthetic_train_portion=None,
            unharden_source=None,
            data_portions=None,
    ):
        super(Unharden_Client, self).__init__(
            learners_ensemble=learners_ensemble,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            test_iterator=test_iterator,
            logger=logger,
            local_steps=local_steps,
            tune_locally=tune_locally
        )

        self.adv_proportion = 0
        self.atk_params = None
        
        # Make copy of dataset and set aside for adv training
        self.og_dataloader = deepcopy(self.train_iterator) # Update self.train_loader every iteration
        
        # Add adversarial client 
        combined_model = self.combine_learners_ensemble()
        self.altered_dataloader = self.gen_customdataloader(self.og_dataloader)
        self.adv_nn = Adv_NN(combined_model, self.altered_dataloader)
        self.dataset_name = dataset_name

        self.synthetic_train_portion = synthetic_train_portion # synthetic data amount in portion of orig data
        self.synthetic = self.synthetic_train_portion != 0.0 # if synthetic data is used

        self.unharden_source = unharden_source # source data used to generate unharden data (orig, synthetic, or orig+synthetic)
        self.poritons_set = (1.0, 1.0, 1.0) # portions of orig, synthetic, and unharden data in final training dataset, sum smaller than 3.0 (orig, synthetic, or unharden)
        self.poritons_set = data_portions # portions of orig, synthetic, and unharden data in final training dataset, sum smaller than 3.0 (orig, synthetic, or unharden)
    
    def gen_customdataloader(self, og_dataloader):
        # Combine Validation Data across all clients as test
        data_x = []
        data_y = []

        for (x,y,idx) in og_dataloader.dataset:
            data_x.append(x)
            data_y.append(y)

        data_x = torch.stack(data_x)
        try:
            data_y = torch.stack(data_y)
        except:
            data_y = torch.tensor(data_y)
        dataloader = Custom_Dataloader(data_x, data_y)
        
        return dataloader
    
    def build_synthetic_data(self):
        if not self.synthetic:
            return
        
        # Build synthetic data
        self.train_iterator.dataset.gen_synthetic_data(self.synthetic_train_portion)
        self.train_iterator.dataset.set_data("synthetic")

        dataloader = self.gen_customdataloader(self.train_iterator)
        # self.synthetic_data = self.train_iterator.dataset.synthetic_data
        self.synthetic_data = dataloader.x_data

        self.adv_nn.synthetize(self.synthetic_data.cuda())
        self.synthetic_target = self.adv_nn.y_syn
        self.train_iterator.dataset.set_synthetic_targets(self.synthetic_target.cpu())
        

    def build_unharden_data(self):
        # Build unharden data
        self.train_iterator.dataset.set_data(self.unharden_source)

        x_adv, y_adv = self.generate_adversarial_data()
        self.unharden_data = x_adv
        self.unharden_target = y_adv
        self.train_iterator.dataset.set_unharden(self.unharden_data.cpu(), self.unharden_target.cpu())

        self.train_iterator.dataset.set_portions(
            orig_portion = self.poritons_set[0],
            synthetic_portion = self.poritons_set[1],
            unharden_portion = self.poritons_set[2]
        )

    def set_adv_params(self, adv_proportion = 0, atk_params = None):
        self.adv_proportion = adv_proportion
        self.atk_params = atk_params
        print("atk_params: ", self.atk_params)
        print("adv_proportion: ", self.adv_proportion)

        self.poritons_set = (1-self.adv_proportion, self.poritons_set[1], self.adv_proportion)
    
    def combine_learners_ensemble(self):

        # This is where the models are stored -- one for each mixture --> learner.model for nn
        hypotheses = self.learners_ensemble.learners

        # obtain the state dict for each of the weights 
        weights_h = []

        model_weights = self.learners_ensemble.learners_weights
        
        for h in hypotheses:
            weights_h += [h.model.state_dict()]
        
        # first make the model with empty weights
        new_model = deepcopy(hypotheses[0].model)
        new_model.eval()
        new_weight_dict = deepcopy(weights_h[0])
        for key in weights_h[0]:
            htemp = model_weights[0]*weights_h[0][key]
            for i in range(1,len(model_weights)):
                htemp+=model_weights[i]*weights_h[i][key]
            new_weight_dict[key] = htemp
        new_model.load_state_dict(new_weight_dict)
        
        return new_model
    
    def update_advnn(self):
        # reassign weights after trained
        self.adv_nn = Adv_NN(self.combine_learners_ensemble(), self.altered_dataloader)
        return
    
    def generate_adversarial_data(self):
        # Generate adversarial datapoints while recognizing idx of sampled without replacement
        
        # Draw random idx without replacement 
        dataloader = self.gen_customdataloader(self.train_iterator)
        x_data = dataloader.x_data
        y_data = dataloader.y_data
        
        self.adv_nn.pgd_sub(self.atk_params, x_data.cuda(), y_data.cuda())
        x_adv = self.adv_nn.x_adv
        y_adv = self.adv_nn.y_adv

        x_adv_res = []
        for i in range(x_adv.shape[0]):
            x_val_normed = x_adv[i]
            try:
                x_val_unnorm = unnormalize_cifar10(x_val_normed)
            except:
                x_val_unnorm = unnormalize_femnist(x_val_normed)

            # x_adv[i] = x_val_unnorm
            x_adv_res.append(x_val_unnorm)
        
        x_adv_res = torch.stack(x_adv_res)
        x_adv = x_adv_res
        
        return x_adv, y_adv
    
    def assign_advdataset(self):
         # Flush current used dataset with original
        self.train_iterator = deepcopy(self.og_dataloader)

        self.build_synthetic_data()
        self.build_unharden_data()

        self.train_loader = iter(self.train_iterator)
        return

    def swap_dataset_labels(self, class_count, switch_pair: bool=True):

        super().swap_dataset_labels(class_count, switch_pair)
        self.og_dataloader = deepcopy(self.train_iterator)
        print("data label swapped, set og_dataloader to new train_iterator")
        