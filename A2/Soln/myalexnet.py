
import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

import torch.nn as nn

# a list of class names
from caffe_classes import class_names


# We modify the torchvision implementation so that the features
# after the final pooling layer is easily accessible by calling
#       net.features(...)
# If you would like to use other layer features, you will need to
# make similar modifications.
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)

        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias

        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.load_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# model_orig = torchvision.models.alexnet(pretrained=True)
model = MyAlexNet()
model.eval()

# Read an image
im = imread('kiwi227.png')[:,:,:3]
im = im - np.mean(im.flatten())
im = im/np.max(np.abs(im.flatten()))
im = np.rollaxis(im, -1).astype(np.float32)

# turn the image into a numpy variable
im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)

# run the forward pass AlexNet prediction
softmax = torch.nn.Softmax()
all_probs = softmax(model.forward(im_v)).data.numpy()[0]
sorted_ans = np.argsort(all_probs)

for i in range(-1, -6, -1):
    print("Answer:", class_names[sorted_ans[i]], ", Prob:", all_probs[sorted_ans[i]])

ans = np.argmax(model.forward(im_v).data.numpy())
prob_ans = softmax(model.forward(im_v)).data.numpy()[0][ans]
print("Top Answer:", class_names[ans], "P(ans) = ", prob_ans)



