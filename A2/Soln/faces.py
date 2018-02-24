from util import *
import torch
from torch.autograd import *


# ----------- CONSTANTS -----------
# We denote male as 1 and female as 0
actor_genders = {'Bracco': 0,
                 'Gilpin': 0,
                 'Harmon': 0,
                 'Baldwin': 1,
                 'Hader': 1,
                 'Carell': 1}
dtype_float = torch.FloatTensor
dtype_long = torch.LongTensor


# ----------- HELPER FUNCTIONS -----------
def process_image(im, resolution = 32):
    """
    Process the given image and output the data
    Args:
        im (str): path to the image
    Returns:
        the processed data
    """
    if resolution == 32:
        path = CROPPED32
    else:
        path = CROPPED64
    data = imread(path + im).flatten() / 225.
    data = data.reshape(resolution * resolution)
    # data = np.insert(data, 0, 1)
    return data.T


def generate_data_dict(resolution = 32):
    """
    Generate the training, validation and test set for all the target actors
    The default test size is 20.
    Note:
    training : val = 80 : 20
    :param resolution: the resolution (32 or 64)
    :type resolution: int
    :return: a dict containing each actor's training, validation and test set
    :rtype: dict
    """
    data = {}
    if resolution == 32:
        path = CROPPED32
    else:
        path = CROPPED64

    for i, actor in enumerate(actor_names):
        all_actor_image = np.array([process_image(image, resolution) for image in os.listdir(path) if actor in image])
        test_size = 20
        train_size = int((all_actor_image.shape[0] - test_size) * .8)
        val_size = all_actor_image.shape[0] - test_size - train_size

        np.random.seed(0)
        np.random.shuffle(all_actor_image)

        data["train" + str(i)] = all_actor_image[:train_size, :]
        data["val" + str(i)] = all_actor_image[train_size: train_size + val_size, :]
        data["test" + str(i)] = all_actor_image[train_size + val_size:, :]

    return data


def get_set(data, type, resolution = 32):
    """
    Get the full train/val/test set
    :param data: the data dict
    :type data: dict
    :param type: the type of set to get (train/val/test)
    :type type: str
    :param resolution: the resolution (32/64)
    :type resolution: int
    :return: the batch data
    :rtype: tuple (x, y)
    """
    batch_x_s = np.zeros((0, resolution * resolution))
    batch_y_s = np.zeros((0, len(actor_names)))
    set_k = [type + str(i) for i in range(len(actor_names))]

    for k in range(len(actor_names)):
        batch_x_s = np.vstack((batch_x_s, ((np.array(data[set_k[k]])[:]) / 255.)))
        one_hot = np.zeros(len(actor_names))
        one_hot[k] = 1
        batch_y_s = np.vstack((batch_y_s, np.tile(one_hot, (len(data[set_k[k]]), 1))))
    return batch_x_s, batch_y_s


def train_nn(x_train, y_train, x_val, y_val, x_test, y_test, dim_h, alpha, epoch, batch_size,
          max_iter, resolution = 32):
    print "========== Start Training =========="
    dim_x = resolution * resolution
    dim_out = 12

    torch.manual_seed(10)
    model = torch.nn.Sequential(torch.nn.Linear(dim_x, dim_h),
                                torch.nn.ReLU(),
                                torch.nn.Linear(dim_h, dim_out))
    # random weights and biases
    model[0].weight = torch.nn.Parameter(torch.randn(model[0].weight.size()))
    model[0].bias = torch.nn.Parameter(torch.randn(model[0].bias.size()))

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    train_results = []
    val_results = []
    test_results = []

    for k in range(epoch):
        print "========== The {}th Epoch ==========".format(k + 1)
        batches = np.array_split(np.random.permutation(range(x_train.shape[0]))[:], batch_size)

        for i, mini_batch in enumerate(batches):
            print "\t===== The {}th Batch =====".format(i + 1)
            x = Variable(torch.from_numpy(x_train[mini_batch]),
                         requires_grad=False).type(dtype_float)
            y_classes = Variable(torch.from_numpy(np.argmax(y_train[mini_batch], 1)),
                                 requires_grad=False).type(dtype_long)
            # Train the model
            for t in range(max_iter):
                y_pred = model(x)
                loss = loss_fn(y_pred, y_classes)

                model.zero_grad()
                loss.backward()
                optimizer.step()

            # train set result
            x = Variable(torch.from_numpy(x_train), requires_grad=False).type(dtype_float)
            y_pred = model(x).data.numpy()
            train_res = np.mean(np.argmax(y_pred, 1) == np.argmax(y_train, 1))
            train_results.append(train_res)

            # val set result
            x = Variable(torch.from_numpy(x_val), requires_grad=False).type(dtype_float)
            y_pred = model(x).data.numpy()
            val_res = np.mean(np.argmax(y_pred, 1) == np.argmax(y_val, 1))
            val_results.append(val_res)

            # test set result
            x = Variable(torch.from_numpy(x_test), requires_grad=False).type(dtype_float)
            y_pred = model(x).data.numpy()
            test_res = np.mean(np.argmax(y_pred, 1) == np.argmax(y_test, 1))
            test_results.append(test_res)

    #Get results on test set
    x = Variable(torch.from_numpy(x_test), requires_grad=False).type(dtype_float)
    y_pred = model(x).data.numpy()
    final_test_result = np.mean(np.argmax(y_pred, 1) == np.argmax(y_test, 1))

    if resolution == 32:
        pickle.dump(model, open("temp/8/model_32.p", "wb"))
    else:
        pickle.dump(model, open("temp/8/model_64.p", "wb"))

    return train_results, val_results, test_results, final_test_result


def weight_visual(W, dir, resolution = 32):

    print W.shape
    for i in range(W.shape[0]):
        img = reshape(W[i, :], (resolution, resolution))
        imsave(dir + "{}.png".format(str(i)), img, cmap = "RdBu")
    return


def part8(resolution = 32):
    data = generate_data_dict(resolution = resolution)
    x_train, y_train = get_set(data, "train", resolution = resolution)
    x_val, y_val = get_set(data, "val", resolution = resolution)
    x_test, y_test = get_set(data, "test", resolution = resolution)

    dim_h = 20
    alpha = 1e-3
    epoch = 5
    batch_size = 5
    max_iter = 1000
    train_results, val_results, test_results, final_test_result = \
        train_nn(x_train, y_train, x_val, y_val, x_test, y_test, dim_h, alpha,
                 epoch, batch_size, max_iter, resolution = resolution)

    print "Final test result: {}".format(final_test_result)

    epochs = np.linspace(1, epoch * batch_size, epoch * batch_size)
    plt.plot(epochs, train_results, 'r-', label= 'Training Result')
    plt.plot(epochs, val_results, 'g-', label= 'Validation Result')
    plt.plot(epochs, test_results, 'b-', label= 'Test Result')
    plt.title('Learning Curve for {}x{}'.format(resolution, resolution))
    plt.xlabel('mini-batch')
    plt.ylabel('accuracy')
    plt.legend(loc="best")
    if resolution == 32:
        plt.savefig("Report/images/8/a.png")
    else:
        plt.savefig("Report/images/8/b.png")
    plt.close()

    return


def part9(resolution = 32):
    if resolution == 32:
        model = pickle.load(open("temp/8/model_32.p", "rb"))
    else:
        model = pickle.load(open("temp/8/model_64.p", "rb"))
    W = model[0].weight.data.numpy()
    weight_visual(W, 'Report/images/9/', resolution)

    # determine which hidden unit is representing which acter
    x = Variable(torch.from_numpy(W), requires_grad=False).type(dtype_float)
    y = model(x).data.numpy()
    for k in range(y.shape[0]):
        print k, actor_names[np.argmax(y[k, :])]


if __name__ == "__main__":
    # part8(32)
    # part9(32)
    part8(64)
    part9(64)
