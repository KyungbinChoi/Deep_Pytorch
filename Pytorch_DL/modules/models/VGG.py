import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, model_code, in_channels, out_dim, act, use_bn, dropout, fc_hid, fc_layers):
        super(VGG, self).__init__()
        
        self.cfg = {
            'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }

        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'leakyrelu':
            self.act = nn.LeakyReLU(0.1)
        else:
            raise ValueError("Not a valid activation function code")
        self.out_dim = out_dim
        self.fc_hid = fc_hid
        self.fc_layers = fc_layers
        self.dropout = dropout
        self.use_bn = use_bn
        self.fc_dropout = nn.Dropout(self.dropout)
        self.layers = self._make_layers(model_code, in_channels, self.use_bn)
        self.fc_layer = self._make_fclayers()

    def _make_layers(self, model_code, in_channels, use_bn):
        layers = []
        for x in self.cfg[model_code]:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]        
            else:
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels, 
                        out_channels=x, 
                        kernel_size=3, 
                        stride=1, 
                        padding=1
                    ),
                    nn.ReLU()
                ]
                if use_bn:
                    layers += [nn.BatchNorm2d(x)]
                layers += [self.act]
                in_channels = x
        
        return nn.Sequential(*layers)

    def _make_fclayers(self):
        fc_layer = []
        fc_layer += [nn.Linear(512, self.fc_hid)]
        fc_layer += [self.act]
        for l in range(self.fc_layers-1):
            fc_layer += [nn.Linear(self.fc_hid, self.fc_hid)]
            fc_layer += [self.act]
            if self.use_bn:
                fc_layer += [nn.BatchNorm1d(self.fc_hid)]
        fc_layer += [nn.Linear(self.fc_hid, self.out_dim)]
        return nn.Sequential(*fc_layer)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0),-1)
        x = self.fc_layer(x)
        return x