import numpy as np
import matplotlib.pyplot as plt

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
    for i in range(0,len(lines)):
        cur = lines[i].strip()
        # need to make sure this line and next line is not empty
        if cur and cur[0] == '>':
            if lines[i+1].strip() and lines[i+1][0] != '>':
                i+=1
                categories[cur[1:]] = [int(x) for x in lines[i].strip().split(",")]
            else:
                categories[cur[1:]] = []
    return categories

def print_dictlist_count(dictlist):
    for key in dictlist:
        print("Count of %s: %d" % (key,len(dictlist[key])))
