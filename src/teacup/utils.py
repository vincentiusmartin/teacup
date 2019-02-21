import numpy as np
import matplotlib.pyplot as plt
import rpy2.robjects as robjects

wilcox = robjects.r['wilcox.test']
numeric = robjects.r['as.numeric']

def wilcox_test(arr1,arr2,alternative='two.sided'):
    narr1 = numeric(arr1)
    narr2 = numeric(arr2)
    #print(wilcox(narr1,narr2,alternative=wilcox_alternative))
    p = wilcox(narr1,narr2,alternative=alternative).rx("p.value")[0][0]
    return p

# =========

def one_index_df(df):
    """
    Moved here from utils. This function makes index df index start from 1
    """
    df.index = np.arange(1, len(df) + 1)
    return df

def dictlist2file(inputdict,filepath):
    with open(filepath,'w') as f:
        for key in inputdict:
            f.write(">%s\n"%key)
            f.write(",".join(str(x) for x in inputdict[key]) + "\n")

def read_dictlist_file(filepath):
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except IOError:
        print("Error: Unable to open file: " + filepath)
        exit(0)

    categories = {}
    i = 0
    while i < len(lines):
        # get the key
        cur = lines[i].strip()
        # need to make sure this line and next line is not empty
        if cur and cur[0] == '>':
            key = cur[1:]
            curlist = []
            i += 1
            while i < len(lines) and lines[i].strip() and lines[i][0] != '>':
                for x in lines[i].strip().split(","):
                    append = float(x) if x != 'NA' else np.NaN
                    curlist.append(append)
                i += 1
            categories[key] = np.array(curlist)
    return categories

def print_dictlist_count(dictlist):
    for key in dictlist:
        print("Count of %s: %d" % (key,len(dictlist[key])))

def scatter_boxplot_dict(groupdict, filepath="scatterbox.png",ylabel=""):
    keys = groupdict.keys()
    listrep = [groupdict[key] for key in keys]

    pos = np.linspace(1,1+len(listrep)*0.5-0.5,len(listrep))
    bp = plt.boxplot(listrep,positions=pos,widths=0.4)
    plt.xticks(pos, keys)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['caps'], color='black')

    for i in range(0,len(listrep)):
        y = listrep[i]
        x = np.random.normal(1+i*0.5, 0.02, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.4,c='red')

    #print("Save distribution of row %s to %s" % (rownum,plotfilename))
    plt.ylabel(ylabel)
    plt.savefig(filepath,positions=[0, 1])
    plt.clf() # clear canvas
