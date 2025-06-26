import torch 
import torch.nn  as nn
import torchvision.transforms.functional as TF


#### UNET ######


class DoubConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels= in_channels, 
                      out_channels= out_channels, 
                      kernel_size= 3,
                      stride= 1,
                      padding= 1,
                      bias=False),

            nn.BatchNorm2d(num_features= out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels= out_channels, 
                      out_channels= out_channels, 
                      kernel_size= 3,
                      stride= 1,
                      padding= 1,
                      bias=False),

            nn.BatchNorm2d(num_features= out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features = [64, 128, 256, 512]):
        super(UNet, self).__init__()

        self.loss_acc = {"train_loss": [], "val_loss": [],
                         "train_acc": [], "val_acc": []}

        self.downs= nn.ModuleList()
        self.ups= nn.ModuleList()
        self.pool= nn.MaxPool2d(kernel_size= 2,
                                stride= 2)
        
        ## down part

        for feature in features:
            self.downs.append(DoubConv(in_channels, feature))
            in_channels= feature

        
        ## up part

        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(
                in_channels= feature*2,
                out_channels= feature,
                kernel_size= 2,
                stride= 2,
                )
            )
            self.ups.append(DoubConv(feature*2, feature))

        self.bottleneck= DoubConv(features[-1], features[-1]*2)
        self.final_conv= nn.Conv2d(
            in_channels=features[0],
            out_channels=out_channels,
            kernel_size=1
        )

    def forward(self, x):
        # print(f"Shape: {x.shape}")
        # print(f'Max: {x.max()} \n Min: {x.min()}')
        skip_connections = []
        for down in self.downs:
            x = down(x)
            # print(f"Shape: {x.shape}")
            # print(f'Max: {x.max()} \n Min: {x.min()}')
            skip_connections.append(x)
            x = self.pool(x)
            # print(f"Shape: {x.shape}")
            # print(f'Max: {x.max()} \n Min: {x.min()}')
        
        x = self.bottleneck(x)
        # print(f"Shape: {x.shape}")
        # print(f'Max: {x.max()} \n Min: {x.min()}')

        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups),2):
            x = self.ups[idx](x)
            # print(f"Shape: {x.shape}")
            # print(f'Max: {x.max()} \n Min: {x.min()}')
            skip_connection = skip_connections[idx // 2]

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x= self.ups[idx +1](concat_skip)
            # print(f"Shape: {x.shape}")
            # print(f'Max: {x.max()} \n Min: {x.min()}')

        return self.final_conv(x)
        
#----------------------------------------------------------------------------------------------#
"""Based on https://github.com/rishikksh20/ResUnet"""

######RESUNET#######


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels, out_channels=1, filters=[64, 128, 256, 512]):
        super(ResUNet, self).__init__()

        self.loss_acc = {"train_loss": [], "val_loss": [],
                         "train_acc": [], "val_acc": []}

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channels, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], out_channels, 1, 1)
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output
 
#----------------------------------------------------------------------------------------------#

######LINEAR PATCHES SEGMENTATION#######

class MultilayerLinear(nn.Module):
    def __init__(self, input_dim=1, size = [128, 64, 32], num_classes=3):
        super().__init__()
        self.loss_acc = {"train_loss": [], "val_loss": [],
                         "train_acc": [], "val_acc": []}
        self.net = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, num_classes)  
            )

    def forward(self, x):
        return self.net(x)




