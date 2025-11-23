"""
    Disclaimer: Implementation based on 'https://dl2.ai/chapter_convolutional-modern/alexnet.html'

"""

from torch import nn
from torchsummary import summary

from utils import ModelConfig

class AlexNet(nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        self.net = nn.Sequential(
            #Convblock 1
            nn.LazyConv2d(96, kernel_size = 11, stride = 4, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            #Convblock 2
            nn.LazyConv2d(256, kernel_size = 5, padding = 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            #Convblock 3
            nn.LazyConv2d(384, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.LazyConv2d(384, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.LazyConv2d(256, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # Flatten to fully connected segment
            nn.Flatten(),
            
            #FC 1
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            #FC 2
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            #Output
            nn.LazyLinear(model_config.num_classes)
        )



    def forward(self, x):
        return self.net(x)


def main():
    model_config = ModelConfig(num_classes = 20)
    model = AlexNet(model_config)
    model.to('cuda')

    summary(model,(3,224,224))

    return 0

if __name__ == '__main__':
    main()
