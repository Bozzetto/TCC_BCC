"""
    Disclaimer: Implementation based on 'https://dl2.ai/chapter_convolutional-modern/resnet.html'
"""

from torch import nn
from torchsummary import summary

from utils import ModelConfig

class Residual(nn.Module):
    
    def __init__(self, conv_channels, use_1x1conv, strides = 1):
        super().__init__()
        
        if use_1x1conv:
            self.direct_conn = nn.Sequential(
                nn.LazyConv2d(conv_channels, kernel_size = 1, padding = 1),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.LazyConv2d(conv_channels, kernel_size = 3, padding = 1),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.LazyConv2d(4*conv_channels, kernel_size = 1, padding = 1),
                nn.LazyBatchNorm2d()
            )
            self.residual_conn = nn.LazyConv2d(4*conv_channels, kernel_size = 1, padding = 1)

        else:
            self.direct_conn = nn.Sequential(
                nn.LazyConv2d(conv_channels, kernel_size = 3, padding = 1),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.LazyConv2d(conv_channels, kernel_size = 3, padding = 1),
                nn.LazyBatchNorm2d(),
            )

            self.residual_conn = nn.LazyConv2d(conv_channels, kernel_size = 1, padding = 1) 

    def forward(self, x):
        y = self.direct_conn(x)
        #x = self.residual_conn(x)
        y += x
        return nn.ReLU(y)

class ResNet(nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        def residual_block(num_residual, conv_channels, use_1x1conv = False):
            layers = []
            for _ in range(num_residual):
                layers.append(Residual(conv_channels, use_1x1conv = use_1x1conv))

            return nn.Sequential(*layers)

        self.start = nn.Sequential(
            nn.LazyConv2d(64, kernel_size = 7, stride = 2),
            nn.LazyBatchNorm2d(),
            nn.MaxPool2d(3, stride = 2)
        )


        match model_config.num_layers:
            case 18:
                self.net = nn.Sequential(
                    residual_block(2, 64),
                    residual_block(2, 128),
                    residual_block(2, 256),
                    residual_block(2, 512)
                )
            case 34:
                self.net = nn.Sequential(
                    residual_block(3, 64),
                    residual_block(4, 128),
                    residual_block(6, 256),
                    residual_block(3, 512)
                )
            case 50:
                self.net = nn.Sequential(
                    residual_block(3, 64, use_1x1conv = True),
                    residual_block(4, 128, use_1x1conv = True),
                    residual_block(6, 256, use_1x1conv = True),
                    residual_block(3, 512, use_1x1conv = True)
                )
            case 101:
                self.net = nn.Sequential(
                    residual_block(3,64, use_1x1conv = True),
                    residual_block(4,128, use_1x1conv = True),
                    residual_block(23,256, use_1x1conv = True),
                    residual_block(3,512, use_1x1conv = True)
                )
        
        self.end = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.LazyLinear(model_config.num_classes)
        )



    def forward(self, x):
        x = self.start(x)
        x = self.net(x)
        return self.end(x)


def main():
    model_config = ModelConfig(num_classes = 20, num_layers = 50)
    model = ResNet(model_config)
    model.to('cuda')

    summary(model,(3,224,224))

    return 0

if __name__ == '__main__':
    main()
