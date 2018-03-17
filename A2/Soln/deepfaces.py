import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable
import numpy as np
import  matplotlib.pyplot as plt
from scipy.misc import imread, imresize
import torch.nn as nn
import os
from util import *

# ========== CONSTANTS ==========
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor


def generate_sets(actor):
    '''Return two lists of randomized image names
    in cropped/ folder that match actor name

    Training Set - At least 67 image names (Screw Peri Gilpin)
    Validation Set - 10 image names
    Testing Set - 10 image names

    Takes in name as lowercase last name (ex: gilpin)
    Assumption: cropped/ folder is populated with images from get_and_crop_images
    '''
    image_list = [f_name for f_name in os.listdir("./Resource/cropped_227/")
                  if actor in f_name]

    np.random.seed(0)
    np.random.shuffle(image_list)

    x_train = np.zeros((0, 3, 227, 227))
    x_val = np.zeros((0, 3, 227, 227))
    x_test = np.zeros((0, 3, 227, 227))

    for i in range(len(image_list)):
        img = imread("./Resource/cropped_227/" + image_list[i])
        img = img[:, :, :3]
        img = img/128. - 1.
        img = np.rollaxis(img, -1).astype(np.float32)
        img = np.reshape(img, [1, 3, 227, 227])
        if i in range(10):
            x_test = np.vstack((x_test, img))
        elif i in range(10, 20):
            x_val = np.vstack((x_val, img))
        else:
            x_train = np.vstack((x_train, img))

    return x_train, x_val, x_test

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
        x = x.view(x.size(0), 256 * 13 * 13)
        return x


def train_nn(dim_h, alpha, max_iter):
    model = MyAlexNet()
    model.eval()
    # hyper-param
    torch.manual_seed(5)
    dim_x = 256 * 13 * 13
    dim_out = len(actor_names)

    # get data
    print "========== Fetching Data =========="
    x_train, y_train = np.zeros((0, 3, 227, 227)), np.zeros((0, len(actor_names)))
    x_val, y_val = np.zeros((0, 3, 227, 227)), np.zeros((0, len(actor_names)))
    x_test, y_test = np.zeros((0, 3, 227, 227)), np.zeros((0, len(actor_names)))

    for i in range(len(actor_names)):
        a_name = actor_names[i]

        x_train_i, x_val_i, x_test_i = generate_sets(a_name)

        one_hot = np.zeros(len(actor_names))
        one_hot[i] = 1

        x_train = np.vstack((x_train, x_train_i))
        x_val = np.vstack((x_val, x_val_i))
        x_test = np.vstack((x_test, x_test_i))

        y_train = np.vstack((y_train, np.tile(one_hot, (x_train_i.shape[0], 1))))
        y_val = np.vstack((y_val, np.tile(one_hot, (x_val_i.shape[0], 1))))
        y_test = np.vstack((y_test, np.tile(one_hot, (x_test_i.shape[0], 1))))

    train_activation = np.zeros((0, 256 * 13 * 13))
    val_activation = np.zeros((0, 256 * 13 * 13))
    test_activation = np.zeros((0, 256 * 13 * 13))

    print x_train.shape
    for i in range(4):
        x = Variable(torch.from_numpy(x_train[100 * i: 100 * (i + 1)]),
                     requires_grad=False).type(dtype_float)
        train_activation = np.vstack((train_activation, model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(x_train[400:]), requires_grad=False).type(dtype_float)
    train_activation = np.vstack((train_activation, model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(x_val), requires_grad=False).type(dtype_float)
    val_activation = np.vstack((val_activation, model.forward(x).data.numpy()))

    x = Variable(torch.from_numpy(x_test), requires_grad=False).type(dtype_float)
    test_activation = np.vstack((test_activation, model.forward(x).data.numpy()))

    # train
    print "========== Start Training =========="
    x = Variable(torch.from_numpy(train_activation), requires_grad=False).type(dtype_float)
    y_classes = Variable(torch.from_numpy(np.argmax(y_train, 1)), requires_grad=False).type(dtype_long)

    new_model = torch.nn.Sequential(
        torch.nn.Linear(dim_x, dim_h),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_h, dim_out),
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    epoch = []
    train_perf = []
    val_perf = []
    test_perf = []


    optimizer = torch.optim.Adam(new_model.parameters(), lr=alpha)
    for i in range(max_iter):
        y_pred = new_model(x)
        loss = loss_fn(y_pred, y_classes)

        new_model.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0 or i == max_iter - 1:
            print "\nIteration {}".format(i)
            epoch.append(i)

            x_train = Variable(torch.from_numpy(train_activation), requires_grad=False).type(dtype_float)
            y_pred = new_model(x_train).data.numpy()
            train_perf_i = (np.mean(np.argmax(y_pred, 1) == np.argmax(y_train, 1))) * 100
            print("Training: {}%".format(train_perf_i))
            train_perf.append(train_perf_i)

            x_val = Variable(torch.from_numpy(val_activation), requires_grad=False).type(dtype_float)
            y_pred = new_model(x_val).data.numpy()
            val_perf_i = (np.mean(np.argmax(y_pred, 1) == np.argmax(y_val, 1))) * 100
            print("Validation: {}%".format(val_perf_i))
            val_perf.append(val_perf_i)

            x_test = Variable(torch.from_numpy(test_activation), requires_grad=False).type(dtype_float)
            y_pred = new_model(x_test).data.numpy()
            test_perf_i = (np.mean(np.argmax(y_pred, 1) == np.argmax(y_test, 1))) * 100
            print("Testing: {}%".format(test_perf_i))
            test_perf.append(test_perf_i)

    return epoch, train_perf, val_perf, test_perf


def part10():
    alpha = 1e-4
    max_iter = 150
    dim_h = 600

    epoch, train_results, val_results, test_results = train_nn(dim_h, alpha, max_iter)


    plt.plot(epoch, train_results, 'r-', label= 'Training Result')
    plt.plot(epoch, val_results, 'g-', label= 'Validation Result')
    plt.plot(epoch, test_results, 'b-', label= 'Test Result')
    plt.title('Learning Curve for {}x{}'.format(227, 227))
    plt.xlabel('mini-batch')
    plt.ylabel('accuracy')
    plt.legend(loc="best")
    plt.savefig("Report/images/10/a.png")
    plt.close()

    return

if __name__ == "__main__":
    part10()
