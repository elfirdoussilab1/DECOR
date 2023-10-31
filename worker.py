import torch
import models
from utils import tools

class Worker(object):
    """ A worker for Decentralized learning (node) """
    def __init__(self, train_data_loader, test_data_loader, batch_size, model, loss, momentum, gradient_clip, sigma_cdp,
                 num_labels, criterion, num_evaluations, device):
        
        # Data loaders
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader

        self.batch_size = batch_size
        self.number_batches = len(train_data_loader)

        self.train_iterator = iter(train_data_loader)
        self.test_iterator = iter(test_data_loader)

        self.device = device
        if self.device == "cuda":
            # Model is on GPU and not explicitly restricted to one particular card => enable data parallelism
            self.model = torch.nn.DataParallel(self.model, device_ids = [0, 1])
        self.loss = getattr(torch.nn, loss)()
        self.model = getattr(models, model)().to(device)
        

        # List of shapes of the model in question, used when unflattening gradients and model parameters
        self.model_shapes = list(param.shape for param in self.model.parameters())
        self.flat_parameters = tools.flatten(self.model.parameters()).to(device)
        self.model_size = len(self.flat_parameters)

        self.momentum = momentum
        self.momentum_gradient = torch.zeros(self.model_size, device=self.device)
        # clip to avoid vanishing and exploiding gradients (or momentums)
        self.gradient_clip = gradient_clip
        
        # sigma_cdp
        self.sigma_cdp = sigma_cdp
        # number of labels (10 on MNIST)
        self.num_labels = num_labels

        # Evaluation criterion
        self.criterion = getattr(tools, criterion)

        # How many evaluations on test batch size to perform during test phase
        self.num_evaluations = num_evaluations

    # Training phase
    def sample_train_batch(self):
        try:
            return next(self.train_iterator)
        except:
            self.train_iterator = iter(self.train_data_loader)
            return next(self.train_iterator)
    
    # Compute gradients on batch = (inputs, targets)
    def compute_gradient(self, X, y):
        self.model.train()
        loss = self.loss(self.model(X), y)
        self.model.zero_grad()
        loss.backward()
        flattened_grad = [param.grad for param in self.model.parameters()]
        return tools.flatten(flattened_grad)
    
    def compute_momentum(self):
        X, y = self.sample_train_batch()
        gradient = self.compute_gradient(X, y)
        if self.momentum > 0: 
            self.momentum_gradient.mul_(self.momentum)
            # mom_gr = (1 - beta)*gr + beta*mom_gr
            self.momentum_gradient.add_((1 - self.momentum) * gradient)
        else:
            self.momentum_gradient = gradient
        if self.gradient_clip is not None:
            self.momentum_gradient = tools.clip_vector(self.momentum_gradient, self.gradient_clip)
        
    def update_model_parameters(self):
        """
        Update model.parameters with values from flat tensor
        """
        unflatened_params = tools.unflatten(self.flat_parameters, self.model_shapes)
        for j, param in enumerate(self.model.parameters()):
            param.data = unflatened_params[j].data.clone().detach()

    # Remark: we can also give him the sum of noises only instead of all noises
    def grad_descent(self, noises, lr, weight_decay):
        """
        noises (Torch.Tensor): A tensor containing the noise term in each row (dim = 0 in sum) !
        lr (float): the learning rate multiplied by the decay 
        """
        # Verification of compatibility of shapes
        assert noises[0].shape == self.momentum_gradient.shape

        self.compute_momentum()
        # Sample noise from normal (0, sigma_cdp^2)
        cdp_noise = torch.normal(mean = torch.zeros_like(self.momentum_gradient), std = self.sigma_cdp**2 )
        self.momentum_gradient.add_(cdp_noise)
        self.momentum_gradient.add_(torch.sum(noises, dim = 0))
        
        # Update parameters
        self.flat_parameters.mul_(1 - lr * weight_decay) 
        self.flat_parameters.add_(-lr * self.momentum_gradient)
        self.update_model_parameters()
        return self.flat_parameters
    

    def decentralized_learning(self, weights, workers_parameters):
        """
        weights (Torch.Tensor): a torch tensor of shape (number of nodes)
        workers_parameters : Tensor containing in each row the flat parameters returned by grad_descent
        """
        # Shape verification
        assert len(weights) == len(workers_parameters)
        self.flat_parameters = torch.sum(weights.view(-1, 1).mul(workers_parameters), dim = 0) # checked !
        self.update_model_parameters()

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
