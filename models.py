# Defining Machine Learning models for experiments
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------- #
# Simple Linear model for MNIST (96 % accuracy)

# Batch_size = 64
# Epochs = 5
class simple_mnist_model(torch.nn.Module):
    def __init__(self):
       super().__init__()
       self.flat = torch.nn.Flatten()
       self.linear = torch.nn.Linear(28*28, 128)
       self.relu = torch.nn.ReLU()
       self.output = torch.nn.Linear(128, 10)    
      
    def forward(self, x):
      x = self.flat(x)
      x = self.linear(x)
      x = self.relu(x)
      logits = self.output(x)
      return logits
    
# ---------------------------------------------------------------------------- #
# Simple convolutional model, for MNIST
class cnn_mnist(torch.nn.Module):
	""" Simple, small convolutional model."""

	def __init__(self):
		""" Model parameter constructor. """
		super().__init__()
		# Build parameters
		self._c1 = torch.nn.Conv2d(1, 20, 5, 1)
		self._c2 = torch.nn.Conv2d(20, 50, 5, 1)
		self._f1 = torch.nn.Linear(800, 500)
		self._f2 = torch.nn.Linear(500, 10)

	def forward(self, x):
		""" Model's forward pass. """
		x = F.relu(self._c1(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self._c2(x))
		x = F.max_pool2d(x, 2, 2)
		x = F.relu(self._f1(x.view(-1, 800)))
		x = F.log_softmax(self._f2(x), dim=1)
		return x

# ---------------------------------------------------------------------------- #
# Simple logistic regression model for MNIST
class logreg_mnist(torch.nn.Module):
	""" Simple logistic regression model."""

	def __init__(self):
		""" Model parameter constructor. """
		super().__init__()
		# Build parameters
		self._linear = torch.nn.Linear(784, 10)

	def forward(self, x):
		""" Model's forward pass. """
		return torch.sigmoid(self._linear(x.view(-1, 784)))

# ---------------------------------------------------------------------------- #
# Logistic regression on LibSVM
class libsvm_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(123, 1)
    
    def forward(self, x):
        x = self.linear(x)
        return torch.sigmoid(x)

# ---------------------------------------------------------------------------- #
#JS: Simple convolutional model, for CIFAR-10/100 (3 input channels)
class cnn_cifar(torch.nn.Module):
  """ Simple, small convolutional model."""

  def __init__(self):
    """ Model parameter constructor."""
    super().__init__()
    # Build parameters
    self._c1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
    self._b1 = torch.nn.BatchNorm2d(self._c1.out_channels)
    self._c2 = torch.nn.Conv2d(self._c1.out_channels, 64, kernel_size=3, padding=1)
    self._b2 = torch.nn.BatchNorm2d(self._c2.out_channels)
    self._m1 = torch.nn.MaxPool2d(2)
    self._d1 = torch.nn.Dropout(p=0.25)
    self._c3 = torch.nn.Conv2d(self._c2.out_channels, 128, kernel_size=3, padding=1)
    self._b3 = torch.nn.BatchNorm2d(self._c3.out_channels)
    self._c4 = torch.nn.Conv2d(self._c3.out_channels, 128, kernel_size=3, padding=1)
    self._b4 = torch.nn.BatchNorm2d(self._c4.out_channels)
    self._m2 = torch.nn.MaxPool2d(2)
    self._d2 = torch.nn.Dropout(p=0.25)
    self._d3 = torch.nn.Dropout(p=0.25)
    self._f1 = torch.nn.Linear(8192, 128)
    self._f2 = torch.nn.Linear(self._f1.out_features, 10)

  def forward(self, x):
    """ Model's forward pass. """
    activation = torch.nn.functional.relu
    flatten    = lambda x: x.view(x.shape[0], -1)
    logsoftmax = torch.nn.functional.log_softmax
    # Forward pass
    x = self._c1(x)
    x = activation(x)
    x = self._b1(x)
    x = self._c2(x)
    x = activation(x)
    x = self._b2(x)
    x = self._m1(x)
    x = self._d1(x)
    x = self._c3(x)
    x = activation(x)
    x = self._b3(x)
    x = self._c4(x)
    x = activation(x)
    x = self._b4(x)
    x = self._m2(x)
    x = self._d2(x)
    x = flatten(x)
    x = self._f1(x)
    x = activation(x)
    x = self._d3(x)
    x = self._f2(x)
    x = logsoftmax(x, dim=1)
    return x


class cifar_Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(20, 200, 5)
        self.fc1 = torch.nn.Linear(200 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x