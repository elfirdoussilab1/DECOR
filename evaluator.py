import torch
import models
import misc

class Evaluator(object):
    """ A worker that is only use to evaluate the global model """
    def __init__(self, train_data_loader_dict, test_data_loader, model, loss,
                 num_labels, criterion, num_evaluations, device):
        
        # Data loaders
        self.train_data_loader_dict = train_data_loader_dict
        self.test_data_loader = test_data_loader

        self.loss = getattr(torch.nn, loss)()
        self.model = getattr(models, model)().to(device)
        self.device = device

        # List of shapes of the model in question, used when unflattening gradients and model parameters
        self.model_shapes = list(param.shape for param in self.model.parameters())
        self.flat_parameters = misc.flatten(self.model.parameters()).to(device)
        self.model_size = len(self.flat_parameters)

        self.num_labels = num_labels

        # Evaluation criterion
        self.criterion = getattr(misc, criterion)

        # How many evaluations on test batch size to perform during test phase
        self.num_evaluations = num_evaluations

        self.num_nodes = len(self.train_data_loader_dict)
    
    def update_model_parameters(self, new_params):
        """
        Update model.parameters with values from new_params
        """
        self.flat_parameters = new_params
        unflatened_params = misc.unflatten(self.flat_parameters, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.data = unflatened_params[j].data.clone().detach()

    def compute_train_loss(self):
        # X and y are the whole training dataset
        train_loss = 0
        for i in range(self.num_nodes):
            # Compute L_i
            train_data_loader = self.train_data_loader_dict[i]
            n_i = len(train_data_loader)
            for X, y in train_data_loader:
                X, y = X.to(self.device), y.to(self.device)
                train_loss += self.loss(self.model(X), y.view(-1, 1)).detach().cpu().numpy() / (n_i * self.num_nodes)
            
        return train_loss
    
    # Test phase
    def sample_test_batch(self):
        try:
            return next(self.test_iterator)
        except:
            self.test_iterator = iter(self.test_data_loader)
            return next(self.test_iterator)
        
    @torch.no_grad()
    def evaluate_model(self):
        """Evaluate the model at the current parameters, with a batch of the given dataset."""
        X, y = self.sample_test_batch()
        X, y = X.to(self.device), y.to(self.device)
        self.model.eval()
        return self.criterion(self.model(X), y)
    
    def compute_accuracy(self):
        accuracy = self.evaluate_model()
        for _ in range(self.num_evaluations - 1):
            accuracy += self.evaluate_model()
        return accuracy[0].item() / accuracy[1].item()