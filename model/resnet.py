"""
    Disclaimer: Implementation based on 'https://dl2.ai/chapter_convolutional-modern/resnet.html'
"""

from torch import nn
from torchsummary import summary

from .utils import ModelConfig

class Residual(nn.Module):
    
    def __init__(self, conv_channels, dimension_expansion, use_1x1conv = False, first_stride = 1):
        super().__init__()
        
        if dimension_expansion:
            self.direct_conn = nn.Sequential(
                nn.LazyConv2d(conv_channels, kernel_size = 1, stride = first_stride),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.LazyConv2d(conv_channels, kernel_size = 3, padding = 1),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.LazyConv2d(4*conv_channels, kernel_size = 1),
                nn.LazyBatchNorm2d()
            )

        else:
            self.direct_conn = nn.Sequential(
                nn.LazyConv2d(conv_channels, kernel_size = 3, stride = first_stride, padding = 1),
                nn.LazyBatchNorm2d(),
                nn.ReLU(),
                nn.LazyConv2d(conv_channels, kernel_size = 3, padding = 1),
                nn.LazyBatchNorm2d(),
            )

        if use_1x1conv:
            shortcut_channels = conv_channels
            if dimension_expansion:
                shortcut_channels *= 4 
            
            self.residual_conn = nn.Sequential(
                nn.LazyConv2d(shortcut_channels, kernel_size = 1, stride = first_stride),
                nn.LazyBatchNorm2d()
            )
        else:
            self.residual_conn = lambda x : x

    def forward(self, x):
        y = self.direct_conn(x)
        x = self.residual_conn(x) 
        y += x
        return nn.functional.relu(y)

class ResNet(nn.Module):

    def __init__(self, model_config: ModelConfig):
        super().__init__()

        def residual_block(num_residual, conv_channels, dimension_expansion = False, stride = 2):
            layers = []
            layers.append(Residual(conv_channels, dimension_expansion, use_1x1conv = True, first_stride = stride))
            
            for _ in range(num_residual - 1):
                layers.append(Residual(conv_channels, dimension_expansion))

            return nn.Sequential(*layers)

        self.start = nn.Sequential(
            nn.LazyConv2d(64, kernel_size = 3, stride = 1) if model_config.small_dataset else nn.LazyConv2d(64, kernel_size = 7, stride = 2),
            nn.LazyBatchNorm2d(),
            nn.Identity() if model_config.small_dataset else nn.MaxPool2d(3, stride = 2)
        )


        match model_config.num_layers:
            case 18:
                self.net = nn.Sequential(
                    residual_block(2, 64, stride = 1),
                    residual_block(2, 128),
                    residual_block(2, 256),
                    residual_block(2, 512)
                )
            case 34:
                self.net = nn.Sequential(
                    residual_block(3, 64, stride = 1),
                    residual_block(4, 128),
                    residual_block(6, 256),
                    residual_block(3, 512)
                )
            case 50:
                self.net = nn.Sequential(
                    residual_block(3, 64, dimension_expansion = True, stride = 1),
                    residual_block(4, 128, dimension_expansion = True),
                    residual_block(6, 256, dimension_expansion = True),
                    residual_block(3, 512, dimension_expansion = True)
                )
            case 101:
                self.net = nn.Sequential(
                    residual_block(3,64, dimension_expansion = True, stride = 1),
                    residual_block(4,128, dimension_expansion = True),
                    residual_block(23,256, dimension_expansion = True),
                    residual_block(3,512, dimension_expansion = True)
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
    model_config = ModelConfig(num_classes = 20, num_layers = 18)
    model = ResNet(model_config)
    model.to('cuda')

    summary(model,(3,224,224))

    return 0

if __name__ == '__main__':
    main()
