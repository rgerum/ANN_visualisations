import numpy as np
import gensim.downloader as api
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

from matplotlib.patches import Circle, Wedge, Polygon
import pylustrator
#pylustrator.start()
##
kdes = []
hists = []
for n in np.arange(1, 10):
    #plt.subplot(3, 3, n-1)
    points = np.random.rand(1000000, n)*2-1
    distance = np.linalg.norm(points, axis=1)
    points = points[distance < 1]
    distance = distance[distance < 1]
    points = points[:100000]
    #print(np.mean(distance < 1))
    print(np.mean(distance < 0.9)/np.mean(distance < 1))
    distance2 = np.linalg.norm(points[:-1:2] - points[1::2], axis=1)
    x = np.arange(0, 2, 0.001)
    #plt.plot(points[:, 0], points[:, 1], "o", ms=.2)
    kdes.append(gaussian_kde(distance))
    hists.append(np.histogram(distance, np.arange(0, 1.01, 0.05)))
    #plt.plot(x, kdes[-1](x))
#    plt.hist(distance2)
    #plt.axis("equal")
    #plt.show()

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
