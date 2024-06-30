import torch
from torch import nn


# block of convolutional layers
def double_conv_block(in_channels: int, out_channels: int, kernel_size):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )


def get_model(input_dim, output_dim, device='cpu'):
    conv_part = nn.Sequential(
        double_conv_block(input_dim[0], 16, 3),
        double_conv_block(16, 32, 3),
        double_conv_block(32, 64, 3),
        nn.Flatten())

    x = torch.unsqueeze(torch.zeros(input_dim), 0)
    size = conv_part.forward(x).size()[1]

    model = nn.Sequential(
        conv_part,
        nn.Linear(size, 256),
        # nn.Dropout(p=0.01),
        nn.ReLU(),
        nn.Linear(256, 64),
        # nn.Dropout(p=0.01),
        nn.ReLU(),
        nn.Linear(64, output_dim)
    )

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.to(device)
    return model


def load_model(file, input_dim, output_dim, device='cpu'):
    model = get_model(input_dim, output_dim, device)
    model.load_state_dict(torch.load(file)['model_state_dict'])
    model.eval()
    return model
