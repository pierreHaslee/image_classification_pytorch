from torch import nn
import torch.nn.functional as F

class PizzaNet(nn.Module):

    def __init__(self, IMG_SIZE):
        super(PizzaNet, self).__init__()

        self.im_size = IMG_SIZE

        self.kernel_size1 = 3
        self.features1 = 128
        self.conv1 = nn.Conv2d(3, self.features1, self.kernel_size1)
        
        self.maxpool1 = nn.MaxPool2d(2,2)

        self.kernel_size2 = 5
        self.features2 = 64
        self.conv2 = nn.Conv2d(self.features1, self.features2, self.kernel_size2)
        
        self.maxpool2 = nn.MaxPool2d(2,2)

        self.kernel_size3 = 5
        self.features3 = 64
        self.conv3 = nn.Conv2d(self.features2, self.features3, self.kernel_size3)
        
        self.maxpool3 = nn.MaxPool2d(2,2)
        
        self.flatten = nn.Flatten()


        self.fc_input_size = int((((self.im_size - self.kernel_size1 + 1)/2 - self.kernel_size2 + 1)/2 - self.kernel_size3 + 1)/2)**2*self.features3
        self.fc1 = nn.Linear(self.fc_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

        self.dropout = nn.Dropout(0.35)
        
        self.soft = nn.Softmax(dim=1)

    def forward(self, batch):

        batch_size = batch.size(0)

        out_conv1 = F.relu(self.conv1(batch))
        out_pool1 = self.maxpool1(out_conv1)
        
        out_conv2 = F.relu(self.conv2(out_pool1))
        out_conv2_drop = self.dropout(out_conv2)
        out_pool2 = self.maxpool2(out_conv2_drop)

        out_conv3 = F.relu(self.conv3(out_pool2))
        out_pool3 = self.maxpool3(out_conv3)

        out_flatten = self.flatten(out_pool3)

        out = F.relu(self.fc1(out_flatten))
        out = self.dropout(out)

        out = F.relu(self.fc2(out))
        out = self.dropout(out)

        out = self.fc3(out)

        out = self.soft(out)

        return out

        