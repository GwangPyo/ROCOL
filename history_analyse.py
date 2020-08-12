import numpy as np
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Process
from multiprocessing import Queue


def __get_bin(hist, q):
    policy_one, policy_zero = discriminate_policy(hist)
    n, _, _ = plt.hist([policy_zero, policy_one], bins=10)
    q.put(n)
    plt.clf()


def get_bin(hist):
    queue = Queue()
    p = Process(target=__get_bin, args=(hist, queue))
    p.start()
    p.join()

    return queue.get()


def discriminate_policy(history):
    policy_zero = []
    policy_one = []
    for h in history:
        if h[1] == 0:
            policy_zero.append(h[0])
        else:
            policy_one.append(h[0])

    print("p_zero mean", np.mean(policy_zero * 10),"num p_zero", len(policy_zero))
    print("p_one mean", np.mean(policy_one * 10), "num_p_one", len(policy_one))
    return policy_one, policy_zero


def load_meta(str_name="histogram_net_action"):
    with open(str_name, "rb") as f:
        history = pickle.load(f)
    bins = get_bin(history)
    bins = np.asarray(bins)
    return bins


def plot_bins(bins):
    zero = bins[0]
    one = bins[1]
    ratio = []
    for z, o in zip(zero, one):
        try:
            ratio.append(z/(z+o))
        except ZeroDivisionError:
            ratio.append(0)
    ratio = np.asarray(ratio)
    plt.bar(np.linspace(0, 10, num=10), ratio, width=1, color='green', label='edge')
    plt.bar(np.linspace(0, 10, num=10), (1 - ratio), bottom=ratio, width=1, color='orange', label='device')
    plt.xlabel("delay")
    plt.ylabel("action selection ratio")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    bins = load_meta("histogram_net_action")
    plot_bins(bins)
