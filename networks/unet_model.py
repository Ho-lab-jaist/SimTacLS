""" Full combination of the parts to establish the complete network """
from .unet_parts import * 

class UNet(nn.Module):
    def __init__(self, n_channels=6, n_classes=4, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Encoder(8, 16)
        self.down2 = Encoder(16, 32)
        self.down3 = Encoder(32, 64)
        self.down4 = Encoder(64, 128)
        self.down5 = Encoder(128, 256)
        self.down6 = nn.Conv2d(256, 256, kernel_size=(3, 2), padding=0)
        factor = 2 if bilinear else 1
        self.up1 = Decoder(256, 128 // factor, bilinear)
        self.up2 = Decoder(128, 64 // factor, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):   # x:  (6, 224, 224)
        x1 = self.inc(x)    # x1: (8, 224, 224)
        x2 = self.down1(x1) # x2: (16, 112, 112)
        x3 = self.down2(x2) # x3: (32, 56, 56)
        x4 = self.down3(x3) # x4: (64, 28, 28)
        x5 = self.down4(x4) # x5: (128, 14, 14)
        x6 = self.down5(x5) # x6: (256, 7, 7)
        x7 = self.down6(x6) # x7: (256, 5, 6)
        x = self.up1(x7, x5)# x:  (128, 10, 12) 
        x = self.up2(x, x4) # x:  (64, 20, 24) 
        force_map = self.outc(x) # force_map:  (3, 20, 24) 
        return force_map

class TacNetUNet(nn.Module):
    def __init__(self, in_nc=6, n_classes=3, num_of_neurons=2048, bilinear=False):
        super(TacNetUNet, self).__init__()
        self.n_channels = in_nc
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_nc, 8)
        self.down1 = Encoder(8, 16)
        self.down2 = Encoder(16, 32)
        self.down3 = Encoder(32, 64)
        self.down4 = Encoder(64, 128)
        self.down5 = Encoder(128, 256)
        self.down6 = nn.Conv2d(256, 256, kernel_size=(3, 2), padding=0)
        factor = 2 if bilinear else 1
        self.up1 = Decoder(256, 128 // factor, bilinear)
        self.up2 = Decoder(128, 64 // factor, bilinear)
        self.outc = OutConv(64, n_classes)

        outputs = 585
        self.fc1 = nn.Linear(1*24*28, num_of_neurons, bias=True)
        self.fc2 = nn.Linear(num_of_neurons, num_of_neurons, bias=True)
        self.fc3 = nn.Linear(num_of_neurons, outputs, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):   # x:  (6, 224, 224)
        x1 = self.inc(x)    # x1: (8, 224, 224)
        x2 = self.down1(x1) # x2: (16, 112, 112)
        x3 = self.down2(x2) # x3: (32, 56, 56)
        x4 = self.down3(x3) # x4: (64, 28, 28)
        x5 = self.down4(x4) # x5: (128, 14, 14)
        x6 = self.down5(x5) # x6: (256, 7, 7)
        x7 = self.down6(x6) # x7: (256, 5, 6)
        x = self.up1(x7, x5)# x:  (128, 10, 12) 
        x = self.up2(x, x4) # x:  (64, 20, 24) 
        force_map = self.outc(x) # force_map:  (3, 24, 28) 
        # feed forward fully connected layers for x feature map (channel)
        output_x = force_map[:, 0, :, :].view(-1, self.num_flat_features(force_map[:, 0, :, :]))
        output_x = self.lrelu(self.fc1(output_x))
        output_x = self.lrelu(self.fc2(output_x))
        output_x = self.fc3(output_x).unsqueeze(1)
        # feed forward fully connected layers for y feature map (channel)
        output_y = force_map[:, 1, :, :].view(-1, self.num_flat_features(force_map[:, 1, :, :]))
        output_y = self.lrelu(self.fc1(output_y))
        output_y = self.lrelu(self.fc2(output_y))
        output_y = self.fc3(output_y).unsqueeze(1)
        # feed forward fully connected layers for y feature map (channel)
        output_z = force_map[:, 2, :, :].view(-1, self.num_flat_features(force_map[:, 2, :, :]))
        output_z = self.lrelu(self.fc1(output_z))
        output_z = self.lrelu(self.fc2(output_z))
        output_z = self.fc3(output_z).unsqueeze(1)
        return torch.cat((output_x, output_y, output_z), dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class TacNetUNet2(nn.Module):
    def __init__(self, in_nc=6, n_classes=3, num_of_neurons=2048, bilinear=False):
        super(TacNetUNet2, self).__init__()
        self.n_channels = in_nc
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(in_nc, 8)
        self.down1 = Encoder(8, 16)
        self.down2 = Encoder(16, 32)
        self.down3 = Encoder(32, 64)
        self.down4 = Encoder(64, 128)
        self.down5 = Encoder(128, 256)
        self.down6 = nn.Conv2d(256, 256, kernel_size=(3, 2), padding=0)
        factor = 2 if bilinear else 1
        self.up1 = Decoder(256, 128 // factor, bilinear)
        self.up2 = Decoder(128, 64 // factor, bilinear)
        self.outc = OutConv(64, n_classes)

        outputs = 585*3
        self.fc1 = nn.Linear(3*24*28, num_of_neurons, bias=True)
        self.fc2 = nn.Linear(num_of_neurons, num_of_neurons, bias=True)
        self.fc3 = nn.Linear(num_of_neurons, outputs, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):   # x:   6, 256, 256
        x1 = self.inc(x)    # x1:  8, 256, 256
        # print(x1.shape)
        x2 = self.down1(x1) # x2: (16, 128, 128)
        # print(x2.shape)
        x3 = self.down2(x2) # x3: (32, 64, 64)
        # print(x3.shape)
        x4 = self.down3(x3) # x4: (64, 32, 32)
        # print(x4.shape)
        x5 = self.down4(x4) # x5: (128, 16, 16)
        # print(x5.shape)
        x6 = self.down5(x5) # x6: (256, 8, 8)
        # print(x6.shape)
        x7 = self.down6(x6) # x7: (256, 6, 7)
        # print(x7.shape)
        x = self.up1(x7, x5)# x:  (128, 12, 14) 
        # print(x.shape)
        x = self.up2(x, x4) # x:  (64, 24, 28) 
        # print(x.shape)
        force_map = self.outc(x) # force_map:  (3, 24, 28) 
        # print(force_map.shape)
        # feed forward fully connected layers for x feature map (channel)
        output = force_map.view(-1, self.num_flat_features(force_map))
        output = self.lrelu(self.fc1(output))
        output = self.lrelu(self.fc2(output))
        output = self.fc3(output)

        return output

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def print_networks(net, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        if verbose:
            print(net)
        print('[TacNet] Total number of parameters : %.3f M' % (num_params / 1e6))
        print('-----------------------------------------------')

if __name__ == '__main__':
    import torch

    def print_networks(net, verbose):
            """Print the total number of parameters in the network and (if verbose) network architecture
            Parameters:
                verbose (bool) -- if verbose: print the network architecture
            """
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print('[TacNet] Total number of parameters : %.3f M' % (num_params / 1e6))
            print('-----------------------------------------------')

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = UNet()
    model.to(dev)
    print_networks(model, False)
    
    # img_fake = torch.rand((2, 6, 224, 224)).to(dev)
    # force_map = model(img_fake)

    # print('input shape:{} - output shape:{}'.format(img_fake.shape, force_map.shape))