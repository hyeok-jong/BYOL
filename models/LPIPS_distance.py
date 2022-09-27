import torch

def normalize_tensor(in_feat,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
    return in_feat/(norm_factor+eps)   # Nan 방지

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

class vgg_distance(torch.nn.Module):
    def __init__(self, model, name):
        super(vgg_distance, self).__init__()
        
        if int(name[3:]) == 11:
            self.layer0 = model[0:2]
            self.layer1 = model[2:5]
            self.layer2 = model[5:10]
            self.layer3 = model[10:15]
            self.layer4 = model[15:20]

        elif int(name[3:]) == 16:
            self.layer0 = model[0:4]
            self.layer1 = model[4:9]
            self.layer2 = model[9:16]
            self.layer3 = model[16:23]
            self.layer4 = model[23:30]

        elif int(name[3:]) == 19:
            self.layer0 = model[0:4]
            self.layer1 = model[4:9]
            self.layer2 = model[9:18]
            self.layer3 = model[18:27]
            self.layer4 = model[27:36]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        return [layer0, layer1, layer2, layer3, layer4]

class resnet_distance(torch.nn.Module):
    def __init__(self, model):
        super(resnet_distance, self).__init__()
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_layer0 = x
        x = self.maxpool(x_layer0)
        x_layer1 = self.layer1(x)
        x_layer2 = self.layer2(x_layer1)
        x_layer3 = self.layer3(x_layer2)
        x_layer4 = self.layer4(x_layer3)

        return [x_layer0, x_layer1, x_layer2, x_layer3, x_layer4]


class alex_distance(torch.nn.Module):
    def __init__(self, model):
        super(alex_distance, self).__init__()
        

        self.layer0 = model[0:2]
        self.layer1 = model[2:5]
        self.layer2 = model[5:8]
        self.layer3 = model[8:10]
        self.layer4 = model[10:12]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        return [layer0, layer1, layer2, layer3, layer4]