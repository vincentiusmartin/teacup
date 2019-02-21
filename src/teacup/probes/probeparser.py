import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt

from teacup import utils

class ProbeParser:
    def __init__(self,probepath,negctrlpath,percentile=95):
        self.cutoff = self.cutoff_from_negctrl(negctrlpath,percentile)

        df = pd.read_csv(probepath,delimiter="\t")
        df.index = np.arange(1, len(df) + 1) # start index from 1

        cols2exp = ['Median', 'Median_o1', 'Median_o2', 'o1_r1', 'o1_r2', 'o1_r3',
           'o2_r1', 'o2_r2', 'o2_r3']
        df = pd.concat([df[['Name']],np.exp(df[cols2exp]),df[['Sequence']]],axis=1)

        m1 = utils.one_index_df(df[df['Name'].str.endswith("m1")].reset_index(drop=True))
        m2 = utils.one_index_df(df[df['Name'].str.endswith("m2")].reset_index(drop=True))
        m3 = utils.one_index_df(df[df['Name'].str.endswith("m3")].reset_index(drop=True))
        wt = utils.one_index_df(df[df['Name'].str.endswith("wt")].reset_index(drop=True))

        self.table = {"m1":m1,"m2":m2,"m3":m3,"wt":wt}
        self.indivsum,self.twosites = self.make_replicas_permutation()
        self.medians = self.get_median_orientation()

    # ======== CONSTRUCTOR FUNCTION ========
    def cutoff_from_negctrl(self,negctrlpath,percentile=95):
        """
        get cutoff from negative control file
        """
        negcutoff = {}
        for orientation in [1,2]:
            colnames = ["o%d_r1" % orientation,"o%d_r2" % orientation,"o%d_r3" % orientation]
            negdf = pd.read_csv(negctrlpath,sep='\t')[colnames].values.tolist()
            flatten = [np.exp(item) for sublist in negdf for item in sublist]
            negcutoff["o%d"%orientation] = np.percentile(flatten,percentile)
        return negcutoff

    def get_median_orientation(self):
        """
        Get only the median column from the table. We have median for orientation 1
        (o1) and orientation 2 (o2). Median from both which is treated as a mean
        value of o1 and o2 denoted as o0.

        return:
            dictionary of [xx]o[y] with xx=m1/m2/m3/wt and y=0/1/2
        """
        med_dict = {}
        orientations = [0,1,2]

        for orientation in orientations:
            if orientation == 0:
                colname = "Median"
            else:
                colname = "Median_o%d" % orientation
            med_dict["m1o%d"%orientation] = self.table['m1'][colname]
            med_dict["m2o%d"%orientation] = self.table['m2'][colname]
            med_dict["m3o%d"%orientation] = self.table['m3'][colname]
            med_dict["wto%d"%orientation] = self.table['wt'][colname]

        return med_dict

    def make_replicas_permutation(self):
        """
        Make permutation for each replicates, useful for hypothesis testing.
        Since individual sites is m1-m3+m2-m3 and two sites is wt - m3, we can take
        m3 from both and have: individual sites = m1 + m2 - m3 and two sites = wt.

        return:
            indivsum = permutations from m1,m2,m3
            twosites = permutations from wt
        """
        indivsum = {}
        twosites = {}

        for orientation in [1,2]:
            replica_cols = ["o%d_r1"%orientation,"o%d_r2"%orientation,"o%d_r3"%orientation]

            indivsum_permut = list(itertools.product(*[replica_cols]*3))
            twosites_permut = list(itertools.product(*[replica_cols]*1))

            indivsum_list = []
            for permut in indivsum_permut:
                indivsum_sum = self.table["m1"][permut[0]] + self.table["m2"][permut[1]] - self.table["m3"][permut[2]]
                indivsum_list.append(indivsum_sum.tolist())
            indivsum["o%d"%orientation] = utils.one_index_df(pd.DataFrame(indivsum_list).transpose())

            twosites_list = []
            for permut in twosites_permut:
                twosites_sum = self.table["wt"][permut[0]]
                twosites_list.append(twosites_sum.tolist())
            twosites["o%d"%orientation] = utils.one_index_df(pd.DataFrame(twosites_list).transpose())

        return indivsum,twosites

    # ======== ANALYSIS FUNCTION ========

    def get_seq(self,seqtype,indexes=[],tofile=False):
        """
        Get sequence of type 'seqtype' and a list containing the desired indexes.
        By default indexes is an emtpy list as if it is empty, we return all
        sequences.
        """
        if not indexes:
            indexes = self.table[seqtype].index.values
        seqlist = self.table[seqtype]['Sequence'][indexes].tolist()
        seqdict = {indexes[i]:seqlist[i] for i in range(len(seqlist))}
        if not tofile:
            return seqdict
        else:
            keys = sorted(seqdict.keys())
            with open("sequences.txt",'w') as f:
                for key in keys:
                    f.write(">%s\n"%key)
                    f.write("%s\n"%seqdict[key])

    def get_mutpos(self,indexes=[]):
        wt = self.table['wt']['Sequence']
        m1 = self.table['m1']['Sequence']
        m2 = self.table['m2']['Sequence']

        if not indexes:
            indexes = wt.index.values

        mutposdict = {}
        for index in indexes:
            wtseq = wt[[index]].iloc[0]
            m1seq = m1[[index]].iloc[0]
            m2seq = m2[[index]].iloc[0]
            diff1 = [u+1 for u in range(len(wtseq)) if wtseq[u] != m1seq[u]]
            diff2 = [u+1 for u in range(len(wtseq)) if wtseq[u] != m2seq[u]]
            mutposdict[index] = diff1+diff2

        return mutposdict

    def scatter_boxplot_permutation(self,rownum):
        """
        make boxplot with scatterplot for individual vs twosites for a row
        """
        for orientation in [1,2]:
            # use loc of iloc as we want to access by index
            indivsum_df = self.indivsum["o%d"%orientation].loc[[rownum]].values[0]
            twosites_df = self.twosites["o%d"%orientation].loc[[rownum]].values[0]

            alldf = [indivsum_df,twosites_df]
            bp = plt.boxplot(alldf,positions = [1,1.5], widths=0.4)
            plt.xticks([1, 1.5], ['individual sum', 'two sites'])
            plt.setp(bp['boxes'], color='black')
            plt.setp(bp['caps'], color='black')

            for i in range(len(alldf)):
                y = alldf[i]
                x = np.random.normal(1+i*0.5, 0.02, size=len(y))
                plt.plot(x, y, 'r.', alpha=0.4,c='red')

            plotfilename = "row%so%d-box.png" % (rownum,orientation)
            print("Save distribution of row %s to %s" % (rownum,plotfilename))
            plt.savefig(plotfilename,positions=[0, 1])
            plt.clf() # clear canvas
        return plotfilename

    def scatter_boxplot_row(self,rownum):
        m1,m2,m3,wt = self.table['m1'], self.table['m2'], self.table['m3'], self.table['wt']
        m1_o1 = m1[['o1_r1','o1_r2','o1_r3']].loc[[rownum]].values[0]
        m2_o1 = m2[['o1_r1','o1_r2','o1_r3']].loc[[rownum]].values[0]
        m3_o1 = m3[['o1_r1','o1_r2','o1_r3']].loc[[rownum]].values[0]
        wt_o1 = wt[['o1_r1','o1_r2','o1_r3']].loc[[rownum]].values[0]
        m1_o2 = m1[['o2_r1','o2_r2','o2_r3']].loc[[rownum]].values[0]
        m2_o2 = m2[['o2_r1','o2_r2','o2_r3']].loc[[rownum]].values[0]
        m3_o2 = m3[['o2_r1','o2_r2','o2_r3']].loc[[rownum]].values[0]
        wt_o2 = wt[['o2_r1','o2_r2','o2_r3']].loc[[rownum]].values[0]

        alldf = [m1_o1,m2_o1,m3_o1,wt_o1,m1_o2,m2_o2,m3_o2,wt_o2]
        pos = np.linspace(1,1+len(alldf)*0.5-0.5,len(alldf))
        bp = plt.boxplot(alldf,positions=pos,widths=0.4)
        plt.xticks(pos, ['m1_o1','m2_o1','m3_o1','wt_o1','m1_o2','m2_o2','m3_o2','wt_o2'])
        plt.setp(bp['boxes'], color='black')
        plt.setp(bp['caps'], color='black')

        for i in range(len(alldf)):
            y = alldf[i]
            x = np.random.normal(1+i*0.5, 0.02, size=len(y))
            plt.plot(x, y, 'r.', alpha=0.4,c='red')

        plotfilename = "row%s-box.png" % (rownum)
        print("Save distribution of row %s to %s" % (rownum,plotfilename))
        plt.savefig(plotfilename,positions=[0, 1])
        plt.clf() # clear canvas
        return plotfilename
