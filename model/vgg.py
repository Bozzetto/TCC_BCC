"""
    Disclaimer: Implementation based on 'https://dl2.ai/chapter_convolutional-modern/vgg.html'

"""

from torch import nn
from torchsummary import summary

from .utils import ModelConfig

class VGG(nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        def vgg_block(num_convs, conv_channels):
            layers = []
            for _ in range(num_convs):
                layers.append(nn.LazyConv2d(conv_channels, kernel_size = 3, padding = 1))
                layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
            return nn.Sequential(*layers)


        match model_config.num_layers:
            case 11:
                self.net = nn.Sequential(
                    vgg_block(1,64),
                    vgg_block(1,128),
                    vgg_block(2,256),
                    vgg_block(2,512),
                    vgg_block(2,512)
                )
            case 13:
                self.net = nn.Sequential(
                    vgg_block(2,64),
                    vgg_block(2,128),
                    vgg_block(2,256),
                    vgg_block(2,512),
                    vgg_block(2,512)
                )
            case 16:
                self.net = nn.Sequential(
                    vgg_block(2,64),
                    vgg_block(2,128),
                    vgg_block(3,256),
                    vgg_block(3,512),
                    vgg_block(3,512)
                )
            case 19:
                self.net = nn.Sequential(
                    vgg_block(2,64),
                    vgg_block(2,128),
                    vgg_block(4,256),
                    vgg_block(4,512),
                    vgg_block(4,512)
                )
        
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.LazyLinear(4096),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.LazyLinear(model_config.num_classes)
        )



    def forward(self, x):
        x = self.net(x)
        return self.linear(x)


def main():
    model_config = ModelConfig(num_classes = 20, num_layers = 16)
    model = VGG(model_config)
    model.to('cuda')

    summary(model,(3,224,224))

    return 0

if __name__ == '__main__':
    main()
