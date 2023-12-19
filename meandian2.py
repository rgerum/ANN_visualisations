import numpy as np

import numpy as np
def meandian(x, n=1.5):
    if not isinstance(n, (tuple, list, np.ndarray)):
        n = np.array([n])
    n = np.asarray(n)
    if np.sum(n<=1):
        meandian1 = brute_meandian2(x, n[n<1])

    meandian2 = interval_meandian(x, n[n>=1])
    meandian = np.zeros(n.shape)
    meandian[n > 1] = meandian2
    meandian[n <= 1] = meandian1
    return meandian


def interval_meandian(x, n=1.5):
    if not isinstance(n, (tuple, list, np.ndarray)):
        n = np.array([n])

    def dp(mu):
        a_mu = x[:, None] - mu[None, :]
        abs_a_mu = np.abs(a_mu)
        abs_a_mu[abs_a_mu < 0.001] = 0.001
        dp = np.sum(-n * abs_a_mu ** (n - 2) * a_mu, axis=0)
        return dp

    interval = np.array([[np.median(x), np.mean(x)]]*len(n)).T
    #interval = np.array([[np.min(x), np.max(x)]]*len(n)).T
    for i in range(100):
        new_point = np.mean(interval, axis=0)
        if np.all(new_point == interval[0]):
            break
        value = dp(new_point)
        interval[((value > 0).astype(np.uint8), np.arange(len(n)))] = new_point
    return np.mean(interval, axis=0)


def brute_meandian2(x, n=1.5):
    if isinstance(n, (np.ndarray, list, tuple)):
        return np.array([brute_meandian2(x, n0) for n0 in n])

    def d(x, mu, n, axis=0):
        x = x[:, None]
        mu = mu[None, :]
        return np.sum((np.abs(x - mu)) ** n, axis=axis)

    v = d(x, x, n)
    i = np.argmin(v, axis=0)
    return x[i]


def lognormal(mu, sigma, size):
    data = np.random.lognormal(mu, sigma, size)
    return data, np.exp(mu - sigma ** 2), np.exp(mu), np.exp(mu + sigma ** 2/2)

def normal(mu, sigma, size):
    data = np.random.normal(mu, sigma, size)
    return data, mu, mu, mu

def poisson(mu, sigma, size):
    data = np.random.poisson(mu, size)
    return data, np.nan, mu + 1/3 - 0.02/mu, mu

def gamma(mu, sigma, size):
    data = np.random.gamma(mu, sigma, size)
    return data, np.nan, np.nan, np.nan

np.random.seed(1234)
mu = 10
sigma = 2
import matplotlib.pyplot as plt
for index, dist in enumerate([normal, lognormal, poisson, gamma]):
    plt.subplot(2, 2, index+1)
    plt.title(dist.__name__)
    for count in [10, 100, 1000]:
        data, exp0, exp1, exp2 = dist(mu, sigma, count)

        n = np.linspace(0.01, 2, 100)
        meand = meandian(data, n)
        l, = plt.plot(n, meand)
        plt.plot([0, 1, 2], [np.nan, np.median(data), np.mean(data)], "+", color=l.get_color())
    plt.plot([0, 1, 2], [exp0, exp1, exp2], "o")
plt.show()
