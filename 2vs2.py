import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylustrator
#pylustrator.start()

def cosine2(x, y, axis=0):
    """ calculate the cosine distance for multidimensional arrays. """
    x = np.asarray(x)
    y = np.asarray(y)
    scalar_product = np.sum(x * y, axis=axis)
    lengthX = np.linalg.norm(x, axis=axis)
    lengthY = np.linalg.norm(y, axis=axis)
    return 1 - scalar_product / (lengthX * lengthY)

def cosine2(x, y, axis=0):
    return np.linalg.norm(x-y, axis=axis)

def two_vs_two(y_test, preds):
    """
    1. Each pair of words is compared only once. takes about 5s. A very naive approach.
    """
    points = 0
    total_points = 0
    for i in range(preds.shape[0] - 1):
        s_i = y_test[i]
        s_i_pred = preds[i]
        for j in range(i + 1, preds.shape[0]):
            s_j = y_test[j]
            s_j_pred = preds[j]

            # Compute cosine distance.
            dsii = cosine2(s_i, s_i_pred)
            dsjj = cosine2(s_j, s_j_pred)
            dsij = cosine2(s_i, s_j_pred)
            dsji = cosine2(s_j, s_i_pred)

            if dsii + dsjj <= dsij + dsji:
                points += 1
            total_points += 1

    return points, total_points, points * 1.0 / total_points


def two_vs_two2(y_test, preds):
    """
    1. Each pair of words is compared only once. takes about 0.1s
    """
    c = cosine2(y_test[:, None, :], preds[None, :, :], axis=2)

    points = 0
    total_points = 0
    # iterate over all pairs
    for i in range(preds.shape[0] - 1):
        for j in range(i + 1, preds.shape[0]):
            # check if the distance from (y1 -> y1_pred + y2 -> y2_pred) < (y1 -> y2_pred + y2 -> y1_pred)
            if c[i, i] + c[j, j] <= c[i, j] + c[j, i]:
                points += 1
            total_points += 1

    return points, total_points, points * 1.0 / total_points


def get_unique_pairs(N):
    a = []
    for i in range(N):
        for j in range(i+1, N):
            a.append([i, j])
    return np.array(a)
    r = np.arange(N)
    xy = np.array(np.meshgrid(r, r)).transpose(1, 2, 0).reshape(-1, 2)
    xy = xy[xy[:, 0] < xy[:, 1]]
    return xy


def two_vs_two3(y_test, preds):
    """
    1. Each pair of words is compared only once. takes about 0.05s. Highly optimized, blazingly fast.
    """
    # get the cosine distance between all points and all predictions
    c = cosine2(y_test[:, None, :], preds[None, :, :], axis=2)
    # get indices for all unique pairs
    i, j = get_unique_pairs(y_test.shape[0]).T
    # calculate the number of points where (y1 -> y1_pred + y2 -> y2_pred) < (y1 -> y2_pred + y2 -> y1_pred)
    points = np.sum(c[i, i] + c[j, j] <= c[i, j] + c[j, i])
    # the the total number of points tested
    total_points = i.shape[0]
    # return the results
    return points, total_points, points * 1.0 / total_points

np.random.seed(1236)
N = 5
f = 2
y = np.random.normal(0, 1, size=(N, f))
y -= np.mean(y, axis=0)
p = 0.3
y2 = y*p+(1-p)*np.random.normal(0, 1, size=(N, f))

y = y[[0, 3, 2, 1, 4]]
y2 = y2[[0, 3, 2, 1, 4]]

current_pair = -1
for current_pair in np.arange(-1, 10):
    for both in [0, 1]:
        ax1 = plt.subplot(121)
        c = np.array(mpl.colors.to_rgba("C0"))
        c = 0.2*c + 0.8 * np.ones(4)
        plt.plot(y[:, 0], y[:, 1], "o", ms=13, mec="C0", color=c, zorder=10, label="ground truth")
        for i, (xx, yy) in enumerate(y):
            plt.text(xx, yy, i, ha="center", va="center", zorder=10)

        if not (current_pair == -1 and both == 0):
            c = np.array(mpl.colors.to_rgba("C1"))
            c = 0.2*c + 0.8 * np.ones(4)
            plt.plot(y2[:, 0], y2[:, 1], "o", ms=13, mec="C1", color=c, zorder=10, label="predicted")
            for i, (xx, yy) in enumerate(y2):
                plt.text(xx, yy, i, ha="center", va="center", zorder=10)
            print(two_vs_two3(y, y2))
        plt.legend()
        pairs = get_unique_pairs(N)
        if current_pair == -1 and both == 1:
            for i in range(N):
                j = i
                def plot(a, b, *args, **kwargs):
                    return plt.plot([y[a, 0], y2[b, 0]], [y[a, 1], y2[b, 1]], *args, **kwargs)
                l1, = plot(i, i, "--", color="gray")

        if current_pair >= 0:
            i, j = pairs[current_pair]
        #for index, pair in enumerate(pairs:

            #i, j = pair
            def plot(a, b, *args, **kwargs):
                return plt.plot([y[a, 0], y2[b, 0]], [y[a, 1], y2[b, 1]], *args, **kwargs)
            l1, = plot(i, i, "-", color=(0., 0., 1., 1))
            l2, = plot(j, j, "-", color=(0, 0, 0.8, 1))
            if both:
                l3, = plot(j, i, "-", color=(1, 0., 0., 1))
                l4, = plot(i, j, "-", color=(0.8, 0., 0., 1))
            else:
                l3, = plt.plot([], [], "-", color=(1, 0., 0., 1))
                l4, = plt.plot([], [], "-", color=(0.8, 0., 0., 1))

            a, b = i, j
            s_i = y[a]
            s_i_pred = y2[a]
            s_j = y[b]
            s_j_pred = y2[b]

            dsii = cosine2(s_i, s_i_pred)
            dsjj = cosine2(s_j, s_j_pred)
            dsij = cosine2(s_i, s_j_pred)
            dsji = cosine2(s_j, s_i_pred)
            valid = dsii + dsjj <= dsij + dsji

                        #f"   ")
            if both:
                legend1 = plt.legend([l1, l2, l3, l4], [f"$d_{{{a}{a}}}$ = {dsii:.2f}", f"$d_{{{b}{b}}}$ = {dsjj:.2f}", f"$d_{{{a}{b}}}$ = {dsij:.2f}", f"$d_{{{b}{a}}}$ = {dsji:.2f}"], loc=0)
            else:
                legend1 = plt.legend([l1, l2, l3, l4], [f"$d_{{{a}{a}}}$ = {dsii:.2f}", f"$d_{{{b}{b}}}$ = {dsjj:.2f}",
                                                        f"$d_{{{a}{b}}}$ = ", f"$d_{{{b}{a}}}$ = "],
                                     loc=0)
            #break
        plt.grid()

        import matplotlib.patches as patches
        ax = plt.subplot(122)
        # Create a Rectangle patch
        plt.text(0.5, len(pairs) + 0.5, "$d_{ii}+d_{jj}$", va="center", ha="center")
        plt.text(1.5, len(pairs) + 0.5, "$d_{ij}+d_{ji}$", va="center", ha="center")
        count = 0
        count2 = 0
        for i, (a,b) in enumerate(pairs):
            print(i, a, b)
            ypos = 9-i
            plt.text(-0.5, ypos+0.5, f"{a, b}", va="center", ha="center")

            s_i = y[a]
            s_i_pred = y2[a]
            s_j = y[b]
            s_j_pred = y2[b]

            # Compute cosine distance.
            dsii = cosine2(s_i, s_i_pred)
            dsjj = cosine2(s_j, s_j_pred)
            dsij = cosine2(s_i, s_j_pred)
            dsji = cosine2(s_j, s_i_pred)
            valid = dsii + dsjj <= dsij + dsji
            if i <= current_pair:
                if valid:
                    count += 1
                else:
                    count2 += 1
                if i == current_pair:
                    print(y[a], y2[a], y[b], y2[b], dsii, dsjj, dsij, dsji)
                    #ax1.set_title(f"$d_{{{a}{a}}}$ = {dsii:.2f} $d_{{{b}{b}}}$ = {dsjj:.2f} $d_{{{a}{b}}}$ = {dsij:.2f} $d_{{{b}{a}}}$ = {dsji:.2f}")

                rect = patches.Rectangle((0, ypos), 1, 1, linewidth=1, edgecolor='k', facecolor='C2' if valid and (both or i != current_pair) else "none", alpha=0.8)
                plt.text(0.5, ypos + 0.5, f"{dsii+dsjj:.2f}", va="center", ha="center")
                ax.add_patch(rect)
                if both or i != current_pair:
                    rect = patches.Rectangle((1, ypos), 1, 1, linewidth=1, edgecolor='k', facecolor='C3' if not valid else "none", alpha=0.8)
                    plt.text(1.5, ypos + 0.5, f"{dsij + dsji:.2f}", va="center", ha="center")
                else:
                    rect = patches.Rectangle((1, ypos), 1, 1, linewidth=1, edgecolor='k', facecolor="none")
                    ax.add_patch(rect)
                ax.add_patch(rect)
                if i == current_pair:
                    rect = patches.Rectangle((0, ypos), 2, 1, linewidth=2, edgecolor=(0, 0, 1), facecolor="none", zorder=20)
                    ax.add_patch(rect)
            else:
                rect = patches.Rectangle((0, ypos), 1, 1, linewidth=1, edgecolor='k', facecolor="none")
                ax.add_patch(rect)
                rect = patches.Rectangle((1, ypos), 1, 1, linewidth=1, edgecolor='k', facecolor="none")
                ax.add_patch(rect)

        if count+count2:
            plt.text(0.5, -1 + 0.5, f"{count}", va="center", ha="center")
            plt.text(1, -1 + 0.5, f":", va="center", ha="center")
            plt.text(1.5, -1 + 0.5, f"{count2}", va="center", ha="center")
            plt.text(2, -1 + 0.5, "$\\rightarrow$", va="center", ha="center")
            plt.text(2.5, -1 + 0.5, f"{count/(count+count2):.2f}", va="center", ha="center")



        plt.xlim(-1, 3)
        plt.ylim(-0.01, 10.01)
        plt.xticks([])
        plt.yticks([])


        plt.figure(1).axes[0].legend(labelspacing=0.7999999999999999, fontsize=10.0, title_fontsize=10.0)
        #plt.figure(1).axes[0].set_position([0.121516, 0.128055, 0.531950, 0.786611])
        plt.figure(1).axes[0].get_legend()._set_loc((0.026273, 0.814606))
        #% start: automatic generated code from pylustrator
        plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
        import matplotlib as mpl
        plt.figure(1).set_size_inches(16.260000/2.54, 10.990000/2.54, forward=True)
        plt.figure(1).axes[0].set_xlim(-1.571864166512109, 1.0)
        plt.figure(1).axes[0].set_ylim(-1.0, 1.3309558433710769)
        plt.figure(1).axes[0].legend(borderpad=0.6, labelspacing=0.7999999999999999, fontsize=10.0, title_fontsize=10.0)
        plt.figure(1).axes[0].set_position([0.121516, 0.128055, 0.531950, 0.786611])
        plt.figure(1).axes[0].get_legend()._set_loc((0.581424, 0.011230))
        plt.figure(1).axes[0].get_legend()._set_loc((0.563800, 0.014173))
        plt.figure(1).axes[0].get_xaxis().get_label().set_text("dimension 1")
        plt.figure(1).axes[0].get_yaxis().get_label().set_text("dimension 2")
        plt.figure(1).axes[1].set_position([0.661397, 0.128055, 0.351234, 0.786611])
        plt.figure(1).axes[1].spines['bottom'].set_visible(False)
        plt.figure(1).axes[1].spines['left'].set_visible(False)
        plt.figure(1).axes[1].spines['right'].set_visible(False)
        plt.figure(1).axes[1].spines['top'].set_visible(False)
        #% end: automatic generated code from pylustrator
        if current_pair >= 0:
            ax1.add_artist(legend1)
        plt.savefig(f"2vs2_{current_pair+1:02}_{both}.png", dpi=300)
        plt.clf()

