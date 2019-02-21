import matplotlib.pyplot as plt
import numpy as np

from teacup import utils

class DNAShape:

    def __init__(self, path):
        self.mgws = utils.read_dictlist_file(path)

    def plot_average(self, labels, path=""):
        plt.clf()
        maxlen = 0
        colors = ['orangered','dodgerblue','lime']
        c = 0

        yall = []
        for label in labels:
            indexes = labels[label]
            curlist = np.array([self.mgws[str(key)] for key in indexes])
            yall.append(curlist)

            seqlen = len(curlist[0])
            maxlen = seqlen
            if not all(len(l) == seqlen for l in curlist):
                raise ValueError('not all lists have same length!')

            ylist = np.median(curlist, axis=0)
            xlist = [i+1 for i in range(0,seqlen)]
            y25p = np.percentile(curlist, 25, axis=0)
            y75p = np.percentile(curlist, 75, axis=0)
            plt.plot(xlist, ylist, alpha=0.8, label=label, c=colors[c], marker='o')
            plt.fill_between(xlist, y75p, y25p, alpha=0.15, facecolor=colors[c])
            c += 1

        pvals = []
        for i in range(0,maxlen):
            # for now assume yall is of size 2
            arr1 = [seq[i] for seq in yall[0]]
            arr2 = [seq[i] for seq in yall[1]]
            if all(np.isnan(x) for x in arr1) and all(np.isnan(x) for x in arr2):
                pvals.append(0)
            else:
                p = utils.wilcox_test(arr1,arr2,"greater")
                #if p < 0.1:
                print(i+1,p)

        plt.xlim(1,maxlen)
        plt.ylabel('Minor groove width')
        plt.xlabel('Sequence position')
        plt.title('Median shape features')
        plt.legend(loc="upper right")
        if not path:
            plt.show()
        else:
            plt.savefig(path)
