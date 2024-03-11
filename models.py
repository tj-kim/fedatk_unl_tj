import torch.nn as nn
import torch.nn.functional as F
import torch 

import torchvision.models as models


class LinearLayer(nn.Module):
    def __init__(self, input_dimension, num_classes, bias=True):
        super(LinearLayer, self).__init__()
        self.input_dimension = input_dimension
        self.num_classes = num_classes
        self.fc = nn.Linear(input_dimension, num_classes, bias=bias)

    def forward(self, x):
        return self.fc(x)


# class FemnistCNN(nn.Module):
#     """
#     Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
#     Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
#     We use `zero`-padding instead of  `same`-padding used in
#      https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
#     """
#     def __init__(self, num_classes):
#         super(FemnistCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 5)

#         self.fc1 = nn.Linear(64 * 4 * 4, 2048)
#         self.output = nn.Linear(2048, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = self.output(x)
#         return x

class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """
    def __init__(self, num_classes):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 4 * 4, 800)
        self.output = nn.Linear(800, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


class CIFAR10CNN(nn.Module):
    def __init__(self, num_classes):
        super(CIFAR10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64 * 5 * 5, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x


# class CelebaCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(CelebaCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
#         # Calculate the output size after convolution and pooling
#         self.fc_input_size = self._get_fc_input_size()
#         self.fc1 = nn.Linear(self.fc_input_size, 2048)
#         self.output = nn.Linear(2048, num_classes)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, self.fc_input_size)
#         x = F.relu(self.fc1(x))
#         x = self.output(x)
#         return x

#     def _get_fc_input_size(self):
#         # Calculate the size of the flattened feature map after convolutions and pooling
#         test_tensor = torch.zeros(1, 3, 55, 45)  # Create a dummy tensor with the desired input shape
#         test_tensor = self.pool(F.relu(self.conv1(test_tensor)))  # Pass through first convolution and pooling
#         test_tensor = self.pool(F.relu(self.conv2(test_tensor)))  # Pass through second convolution and pooling
#         return test_tensor.view(-1).shape[0]  # Return the size of the flattened feature map

import torch
import torch.nn as nn
import torch.nn.functional as F

class CelebaCNN(nn.Module):
    def __init__(self, num_classes):
        super(CelebaCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(16)  # Batch normalization after first convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(32)  # Batch normalization after second convolutional layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3_bn = nn.BatchNorm2d(64)  # Batch normalization after third convolutional layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Calculate the output size after convolution and pooling
        self.fc_input_size = self._get_fc_input_size()
        self.fc1 = nn.Linear(self.fc_input_size, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))  # Apply batch normalization after the first convolutional layer
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))  # Apply batch normalization after the second convolutional layer
        x = self.pool(F.relu(self.conv3_bn(self.conv3(x))))  # Apply batch normalization after the third convolutional layer
        x = x.view(-1, self.fc_input_size)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

    def _get_fc_input_size(self):
        # Calculate the size of the flattened feature map after convolutions and pooling
        test_tensor = torch.zeros(1, 3, 55, 45)  # Create a dummy tensor with the desired input shape
        test_tensor = self.pool(F.relu(self.conv1_bn(self.conv1(test_tensor))))  # Pass through first convolution, batch normalization, and pooling
        test_tensor = self.pool(F.relu(self.conv2_bn(self.conv2(test_tensor))))  # Pass through second convolution, batch normalization, and pooling
        test_tensor = self.pool(F.relu(self.conv3_bn(self.conv3(test_tensor))))  # Pass through third convolution, batch normalization, and pooling
        return test_tensor.view(-1).shape[0]  # Return the size of the flattened feature map



def getCelebaCNN(n_classes):
    """
    creates VGG11 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = CelebaCNN(n_classes)
    # model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)

    return model

class NextCharacterLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers):
        super(NextCharacterLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, embed_size)

        self.rnn =\
            nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True
            )

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input_):
        encoded = self.encoder(input_)
        output, _ = self.rnn(encoded)
        output = self.decoder(output)
        output = output.permute(0, 2, 1)  # change dimension to (B, C, T)
        return output


def get_vgg11(n_classes):
    """
    creates VGG11 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.vgg11(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, n_classes)

    return model


def get_squeezenet(n_classes):
    """
    creates SqueezeNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.squeezenet1_0(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, n_classes, kernel_size=(1, 1), stride=(1, 1))
    model.num_classes = n_classes

    return model


def get_mobilenet(n_classes):
    """
    creates MobileNet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)

    return model


def get_resnet18(n_classes):
    """
    creates Resnet model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model


def get_resnet34(n_classes):
    """
    creates Resnet34 model with `n_classes` outputs
    :param n_classes:
    :return: nn.Module
    """
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    return model
