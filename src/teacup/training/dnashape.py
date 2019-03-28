import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.backends.backend_pdf import PdfPages

from teacup import utils
from teacup.training import trainingparser

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

    def get_shape_names():
        return ["ProT","MGW","Roll","HelT"]

    def plot_average(self, labels, pthres=0.05, mark=[], path=".", plotlabel="Average DNA shape"):
        plt.clf()
        colors = ['orangered','dodgerblue','lime']

        shapes = {"Propeller twist ":self.prot,"Helix twist":self.helt,"Roll":self.roll,"Minor Groove Width":self.mgw}
        #shapes = {"Propeller twist ":self.prot}
        keystolabel = {"additive":"ã","cooperative":"ç"}

        fig = plt.figure(figsize=(12,12))
        #with PdfPages("shape.pdf") as pdf:
        n = 0
        for sh in shapes:
            c = 0
            n += 1
            ax = fig.add_subplot(2,2,n)
            yall = {}

            # ==== Plotting part using 25, 50, and 75 percentiles ====
            # labels = cooperative or additive
            keys = []
            for label in labels:
                indexes = labels[label]
                curlist = np.array([shapes[sh][str(key)] for key in indexes])
                keys.append(label) # for marking significance
                yall[label] = curlist

                seqlen = len(curlist[0])
                if not all(len(l) == seqlen for l in curlist):
                    raise ValueError('not all lists have same length!')

                ylist = np.median(curlist, axis=0)
                xlist = [i+1 for i in range(0,seqlen)]
                y25p = np.percentile(curlist, 25, axis=0)
                y75p = np.percentile(curlist, 75, axis=0)
                ax.plot(xlist, ylist, alpha=0.8, label=label, c=colors[c], marker='o')
                ax.fill_between(xlist, y75p, y25p, alpha=0.15, facecolor=colors[c])
                c += 1

            # ==== Hypothesis testing to mark significant binding sites ====
            signiflabel = []
            for i in range(0,seqlen):
                # for now assume yall is of size 2
                arr_coop = [seq[i] for seq in yall['cooperative']]
                arr_add = [seq[i] for seq in yall['additive']]
                if not (all(np.isnan(x) for x in arr_coop) and all(np.isnan(x) for x in arr_add)):
                    p1 = utils.wilcox_test(arr_coop,arr_add,"greater")
                    p2 = utils.wilcox_test(arr_coop,arr_add,"less")
                    if p1 <= pthres:
                        signiflabel.append(keystolabel["cooperative"])
                    elif p2 <= pthres:
                        signiflabel.append(keystolabel["additive"])
                    else:
                        signiflabel.append('')
                else:
                    signiflabel.append('')

            # ==== Mark binding sites as given from the input ====
            for m in mark:
                ax.axvline(x=m,linewidth=1, color='g',linestyle='--')

            ax.set_xlim(1,seqlen)
            xi = [x for x in range(1,seqlen+1)]
            #label = ['' if x not in signiflist else '*' for x in xi]
            ax.set_xticks(xi)
            ax.set_xticklabels(signiflabel)
            ax.yaxis.set_label_text(sh)
            ax.xaxis.set_label_text('Sequence position')
            ax.legend(loc="upper right")
            ax.set_title(plotlabel)

        with PdfPages(path) as pdf:
            pdf.savefig(fig)

# ============== This is a separate class to contain everything ==============

class DNAShapes:

    def __init__(self, path, bsites):
        # INITIALIZE
        self.shapedict = {}
        self.dists = []
        self.bsites = bsites

        dirs = next(os.walk(path))[1]
        self.dists = [int(d[1:]) for d in dirs]
        self.maxdist = max(self.dists)
        for distdir in dirs:
            distpath = "%s/%s" % (path,distdir)
            shape = DNAShape(distpath)
            self.shapedict[distdir] = shape

    def get_features(self):
        span_out = 3
        numseq = len(self.bsites[0])
        shape_names = DNAShape.get_shape_names()
        features = {name:[[] for _ in range(numseq)] for name in shape_names}

        for dist,shape_dist in self.shapedict.items():
            for shape_name in shape_names:
                shapes = getattr(shape_dist, shape_name.lower())
                for seqid in shapes:
                    print(len(shapes[seqid]))
                break
            break

    # TODO FIX BASED ON THE CLASS
    def plot_average_all(trainingpath,shapepath,distances):
        for dist in distances:
            print("Plotting for dist %d" % dist)
            dist_path = "%s/d%s" % (shapepath,dist)
            bpos_file = "%s/bindingpos.txt" % dist_path
            with open(bpos_file,'r') as f:
                bpos = f.read().strip().split(",")
                bpos = [int(x) for x in bpos]
            train = trainingparser.TrainingParser(trainingpath,motiflen=6)
            # make a new data frame with only the distance on each iteration
            t2 = train.training.loc[train.training['distance'] == dist]
            train2 = trainingparser.TrainingParser(t2,motiflen=6)
            li = train2.get_labels_indexes()
            shape = DNAShape(dist_path)
            for p in [0.05,0.1]:
                plot_path = "%s/shape-p=%.2f.pdf"%(dist_path,p)
                shape.plot_average(li,pthres=p,mark=bpos,path=plot_path,plotlabel="Average DNA shape for d=%d,p=%.2f" % (dist,p))
