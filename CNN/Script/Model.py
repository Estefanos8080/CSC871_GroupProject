import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, hyperparameters):
        super(Model, self).__init__()
        self.hyperparameters = hyperparameters

        if hyperparameters['architecture'] == 'CNN':
            self.architecture = CNN(hyperparameters)

    def forward(self, x):
        return self.architecture(x)


class CNN(nn.Module):
    def __init__(self, hyperparameters):
        super(CNN, self).__init__()
        self.filters = hyperparameters['filters']
        self.layers = hyperparameters['layers']
        self.dropout_rate = hyperparameters['dropout_rate']

        self.conv_layers = self._create_conv_layers()

        # Call _calculate_to_linear to compute _to_linear
        self._calculate_to_linear()

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def _create_conv_layers(self):
        conv_layers = []
        in_channels = 3
        for _ in range(self.layers):
            conv_layers.extend([
                nn.Conv2d(in_channels, self.filters, kernel_size=7, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(self.filters),
                nn.Dropout(self.dropout_rate)
            ])
            in_channels = self.filters
        return nn.Sequential(*conv_layers)

    def _calculate_to_linear(self):
        with torch.no_grad():
            # will be replaced by the actual tensors of the datasampels
            dummy_x = torch.zeros(1, 3, 187, 187)
            dummy_x = self.conv_layers(dummy_x)
            self._to_linear = int(dummy_x.view(dummy_x.size(0), -1).shape[1])

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return torch.sigmoid(x)

# Hardcodeed
hyperparameters = {
    'architecture': 'CNN',
    'filters': 64,
    'layers': 4,
    'dropout_rate': 0.3
}

model = Model(hyperparameters)
print(model)
