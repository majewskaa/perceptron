from neural import DlNet, f
import numpy as np
import matplotlib.pyplot as plt

L_BOUND = -5
U_BOUND = 5
EPOCHS = 10_000
HIDDEN_NEURONS = 16
MINI_BATCH_SIZE = 100
LEARNING_RATE = 0.1

def loss(result, expected):
    return np.square(expected - result)


if __name__ == "__main__":
    x = np.linspace(L_BOUND, U_BOUND, 1000, dtype=np.double)
    y = f(x)

    nn = DlNet(HIDDEN_NEURONS, LEARNING_RATE)
    nn.train_gif(x, y, EPOCHS, MINI_BATCH_SIZE)

    yh = [nn.predict(xn) for xn in x]

    n_loss = loss(yh, y).mean()

    print(f'Loss: {n_loss}')

    plt.plot(x,y, 'r')
    plt.plot(x,yh, 'b')
    plt.savefig('final_plot.png')
