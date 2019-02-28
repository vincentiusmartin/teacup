import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.backends.backend_pdf import PdfPages

from teacup import utils

class DNAShape:

    def __init__(self, path): #bpos1_list, bpos2_list
        # TODO: Checking in here
        for root, dirs, files in os.walk(path):
            for filename in files:
                path = "%s/%s" % (root,filename)
                filename, file_extension = os.path.splitext(path)
                if file_extension == ".ProT":
                    self.prot = utils.read_dictlist_file(path)
                elif file_extension == ".MGW":
                    self.mgw = utils.read_dictlist_file(path)
                elif file_extension == ".Roll":
                    self.roll = utils.read_dictlist_file(path)
                elif file_extension == ".HelT":
                    self.helt = utils.read_dictlist_file(path)

    def plot_average(self, labels, pthres=0.05, path="", mark=[]):
        plt.clf()
        maxlen = 0
        colors = ['orangered','dodgerblue','lime']

        shapes = {"Propeller twist ":self.prot,"Helix twist":self.helt,"Roll":self.roll,"Minor Groove Width":self.mgw}

        yall = []
        fig = plt.figure(figsize=(12,12))
        #with PdfPages("shape.pdf") as pdf:
        n = 0
        for sh in shapes:
            c = 0
            n += 1
            ax = fig.add_subplot(2,2,n)
            for label in labels:
                indexes = labels[label]
                curlist = np.array([shapes[sh][str(key)] for key in indexes])
                yall.append(curlist)

                seqlen = len(curlist[0])
                maxlen = seqlen
                if not all(len(l) == seqlen for l in curlist):
                    raise ValueError('not all lists have same length!')

                ylist = np.median(curlist, axis=0)
                xlist = [i+1 for i in range(0,seqlen)]
                y25p = np.percentile(curlist, 25, axis=0)
                y75p = np.percentile(curlist, 75, axis=0)
                ax.plot(xlist, ylist, alpha=0.8, label=label, c=colors[c], marker='o')
                ax.fill_between(xlist, y75p, y25p, alpha=0.15, facecolor=colors[c])
                c += 1

            signiflist = []
            for i in range(0,maxlen):
                # for now assume yall is of size 2
                arr1 = [seq[i] for seq in yall[0]]
                arr2 = [seq[i] for seq in yall[1]]
                if not (all(np.isnan(x) for x in arr1) and all(np.isnan(x) for x in arr2)):
                    p1 = utils.wilcox_test(arr1,arr2,"greater")
                    p2 = utils.wilcox_test(arr1,arr2,"less")
                    if p1 <= pthres or p2 <= pthres:
                        signiflist.append(i+1)

            for m in mark:
                ax.axvline(x=m,linewidth=1, color='g',linestyle='--')

            ax.set_xlim(1,maxlen)
            xi = [x for x in range(1,maxlen)]
            label = ['' if x not in signiflist else '*' for x in xi]
            ax.set_xticks(xi)
            ax.set_xticklabels(label)
            ax.yaxis.set_label_text(sh)
            ax.xaxis.set_label_text('Sequence position')
            ax.legend(loc="upper right")

        with PdfPages("shape-p=%.2f.pdf"%pthres) as pdf:
            pdf.savefig(fig)
