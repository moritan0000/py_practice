from chainer import Chain, Variable, Link
from chainer.links import Linear
from chainer.functions import relu, softmax_cross_entropy
from chainer.optimizers import AdaGrad
from chainer.datasets import get_mnist
import numpy as np

n_nodes = 100
n_iter = 100
batch_size = 100


class NeuralNetwork(Chain):
    def __init__(self, n_units, n_out):
        super().__init__(l1=Linear(None, n_units),
                         l2=Linear(n_units, n_units),
                         l3=Linear(n_units, n_out))

    def __call__(self, x):
        h1 = relu(self.l1(x))
        h2 = relu(self.l2(h1))
        return self.l3(h2)


def calc_accuracy(model, xs, ts):
    ys = model(xs)
    loss = softmax_cross_entropy(ys, ts)
    ys = np.argmax(ys.data, axis=1)
    cors = (ys == ts)
    num_cors = sum(cors)
    accuracy = num_cors / ts.shape[0]
    return accuracy, loss


def main():
    model = NeuralNetwork(n_nodes, 10)
    optimizer = AdaGrad()
    optimizer.setup(model)

    train, test = get_mnist()
    xs, ts = train._datasets
    print(xs[0].shape)
    print(ts[0])
    txs, tts = test._datasets

    for i in range(n_iter):
        for j in range(600):
            model.cleargrads()
            x = xs[(j * batch_size):((j + 1) * batch_size)]
            t = ts[(j * batch_size):((j + 1) * batch_size)]
            t = Variable(np.array(t, "i"))
            y = model(x)
            loss = softmax_cross_entropy(y, t)
            loss.backward()
            optimizer.update()

        accuracy_train, loss_train = calc_accuracy(model, xs, ts)
        accuracy_test, _ = calc_accuracy(model, txs, tts)

        print("Epoch {}: Acc.(train) = {:.4f}, Acc.(test) = {:.4f}".format(
            i + 1, accuracy_train, accuracy_test))


main()
