import numpy as np
import gensim.downloader as api
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from matplotlib.patches import Circle, Wedge, Polygon
import pylustrator
import matplotlib as mpl
pylustrator.start()
##

def norm(points, order, axis=1):
    return np.sum(np.abs(points)**order, axis=axis)**(1/order)

def inner_mean(points, order):
    distance = norm(points, order)
    points = points[distance < 1]
    return distance[distance < 1]

kdes = []
hists = []
relative = []
ns = []
for n in np.geomspace(1, 1000, 10):
    n = int(n)
    #plt.subplot(3, 3, n-1)
    #points = np.random.rand(100000, n)*2-1
    points = np.random.normal(0, 1, size=(100000, n))
    distance2 = np.linalg.norm(points[:-1:2] - points[1::2], axis=1)
    mini, maxi = np.percentile(distance2, [1, 99])
    relative.append(mini/maxi)
    ns.append(n)
    continue
    print(n, np.percentile(distance2, [1, 99]))
    distance2 -= np.mean(distance2)

    d, b = np.histogram(distance2, np.linspace(0, np.max(distance2), 1000), density=True)
    #plt.plot(x, kdes[-1](x), label=f"{n}D")
    f = 1 - (n-1) /(1000-1)
    color = np.array(mpl.colors.to_rgb("C0")) * f + np.array(mpl.colors.to_rgb("C3")) * (1-f)
    plt.plot(b[:-1], d, label=f"{n}D", color=color)
#    plt.hist(distance2)
    #plt.axis("equal")
plt.plot(ns, relative)
plt.fill_between(ns, relative, np.ones(len(relative)), color="C0", alpha=0.5)
plt.axhline(1, color="k", lw=0.8)
plt.xlabel("distance")
plt.legend()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
plt.figure(1).set_size_inches(7.980000/2.54, 6.000000/2.54, forward=True)
plt.figure(1).axes[0].set_position([0.175955, 0.198983, 0.704936, 0.770000])
plt.figure(1).axes[0].spines['right'].set_visible(False)
plt.figure(1).axes[0].spines['top'].set_visible(False)
plt.figure(1).axes[0].get_legend()._set_loc((0.831604, 0.401093))
plt.figure(1).axes[0].get_yaxis().get_label().set_text("density")
#% end: automatic generated code from pylustrator
plt.savefig("distance_high_dimensions_neighbors.png")
plt.show()

if 0:
    plt.cla()
    for n in np.arange(1, 10):
        x = np.arange(0, 2, 0.01)
        d = kdes[n-1](x)
        d /= np.max(d)
        for i, d0 in zip(x, d):
            plt.gca().add_patch(Wedge((0, 0), i, 30*n, 30*(n+1), width=0.01, color="C1", alpha=d0, linewidth=0))
        plt.xlim(-1., 1)
        plt.ylim(-1., 1)

if 0:
    plt.cla()
    import matplotlib as mpl
    im = plt.imread("Orange.png")
    plt.figure(0, (4, 4))
    plt.axes([0, 0, 1, 1])
    plt.imshow(im, extent=[-1.1, 1.1, -1.1, 1.1])
    for n in np.arange(1, 6):
        d, x = hists[n-1]
        d = d / np.max(d)
        dx = x[1]-x[0]
        for i, d0 in zip(x[:-1], d):
            color = d0*np.array(mpl.colors.to_rgb("#f8aa00")) + (1-d0) * np.ones(3)

            plt.gca().add_patch(Wedge((0, 0), i+dx, 180-36*(n), 180-36*(n-1),  width=dx, color=color, linewidth=1))

        angle = np.deg2rad(180-36*(n-0.5))
        angle0 = np.deg2rad(180-36*(n-1))
        angle1 = np.deg2rad(180-36*(n-0))
        r = 1.15
        plt.text(np.cos(angle)*r, np.sin(angle)*r, f"{n}D", va="center", ha="center", fontsize=12)
        r = 1.1
        plt.plot([0, np.cos(angle0)*r], [0, np.sin(angle0)*r], lw=1.4, color="#c1ba9b")
        plt.plot([0, np.cos(angle1)*r], [0, np.sin(angle1)*r], lw=1.4, color="#c1ba9b")
        r = 1.0
        plt.plot([0, np.cos(angle0)*r], [0, np.sin(angle0)*r], lw=1.5, color="#f5edcd")
        plt.plot([0, np.cos(angle1)*r], [0, np.sin(angle1)*r], lw=1.5, color="#f5edcd")
        plt.xlim(-1.3, 1.3)
        plt.ylim(-1.3, 1.3)

    plt.gca().axis('off')
    plt.savefig("distance_high_dimensions_orange.png", dpi=150)

    plt.show()

if 0:
    ##
    exit()


    def cosine2(x, y, axis=0):
        """ calculate the cosine distance for multidimensional arrays. """
        x = np.asarray(x)
        y = np.asarray(y)
        scalar_product = np.sum(x*y, axis=axis)
        lengthX = np.linalg.norm(x, axis=axis)
        lengthY = np.linalg.norm(y, axis=axis)
        return 1 - scalar_product / (lengthX * lengthY)

    word_vectors = api.load('word2vec-google-news-300')

    words = word_vectors.key_to_index

    word_group = word_vectors[[i for i in range(1000)]]

    distance = np.linalg.norm(word_group - bad, axis=1)
    distanceCos = cosine2(word_group, bad[None, :], axis=1)

    bad = word_vectors["bad"]
    good = word_vectors["good"]
    house = word_vectors["house"]
    np.linalg.norm(bad-good)
    print(cosine2(bad, good))

    print(np.linalg.norm(bad-house))
    print(cosine2(bad, house))
