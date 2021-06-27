from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
from OC.config import nruns


def data(path):
    f = open(path, 'r', encoding='utf-8')


def plot(history):
    for run in range(nruns):
        clear_output(True)
        plt.figure(figsize=(20, 6))
        plt.subplot(121)
        plt.title('run: %s' % run)
        plt.xlabel('episodes')
        plt.ylabel('steps')
        plt.plot(np.mean(history[:run + 1, :, 0], axis=0))
        plt.grid(True)
        plt.subplot(122)
        plt.title('run: %s' % run)
        plt.xlabel('episodes')
        plt.ylabel('avg. option duration')
        plt.plot(np.mean(history[:run + 1, :, 1], axis=0))
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    f = open()
    print("1")
