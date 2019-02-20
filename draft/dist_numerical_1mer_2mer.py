# This script adds kmer features to the existing data

import pandas as pd
import itertools

NEW_DATA_PATH = "../data/generated/training_distance_1mer_2mer.csv"
DATA_PATH = "../data/generated/training.csv"

def get_seq(data):
    '''
    This function returns the list of sequences from the data
    in a numpy array
    Input: input data as a dataframe
    Output: DNA sequences in the data in a numpy array
    '''

    return data["sequence"].values

def get_kmer(num, seq, bpos1, bpos2):
    '''
    Number of distinct k-mer in the sequence
    Input: num, seq where num = k and seq is the sequence
           to be evaluated
    Output: len(kmer_set) which is the number of distinct k-mers
            in the sequence
    '''

    # initialize an empty dict
    alph = ['A', 'G', 'C', 'T']
    kmer_dict = {}
    # create a dictionary for kmer
    for perm in itertools.product(alph, repeat=num):
        kmer_dict[''.join(perm)] = 0

    # initialize index i to 0
    i = 0
    #iterate through the sequence
    while bpos1 + 4 + i < bpos2 - 2:
        # get the read of length num
        read = seq[bpos1+4+i:bpos1+4+i+num]
        # add this read to the set
        kmer_dict[read] += 1
        # increment index
        i += 1

    # return the dictionary
    return kmer_dict

def get_col(num, seq_col, bpos1_col, bpos2_col):
    # initialize an empty list to store the output
    comb_output = []
    
    # iterate through each sequence in the data
    for seq, bpos1, bpos2 in zip(seq_col, bpos1_col, bpos2_col):
        # append the number of distinct k-mer in the sequence
        # to the output list
        output = []
        # ret is a dictionary of count of every possible num-mer
        ret = get_kmer(num, seq, bpos1, bpos2)
        for val in ret.values():
            output.append(val)
        comb_output.append(output)
    # return the output
    return ret, comb_output

def add_col(df, num, kmer_dict, comb_output):
    '''
    This function inserts the new column into the dataframe
    Input: 
    Output: df which is the updated dataframe
    '''

    newDf = []
    label = df['label'].values
    df = df.drop(columns=['label'], axis=1)

    # insert each row
    for i in range(len(comb_output)):
        # insert the new column's name in the dataframe
        newRow = list(df.iloc[i].values) + comb_output[i] + [label[i]]
        newDf.append(newRow)

    col_list = []
    # add new column names
    for key in kmer_dict.keys():
        col_name = str(num) +"mer_" + key + "_count"
        col_list.append(col_name)
    columns = list(df.columns) + col_list + ['label']

    # return a new dataframe
    return pd.DataFrame(newDf, columns=columns)

def make_categorical(df):
    # replace numerical distance with categorical
    one_hot = pd.get_dummies(df['distance']).values.tolist()
    col_names = []
    df = df.drop(['distance'], axis=1)
    label = df['label'].values
    df = df.drop(['label'], axis=1)

    newDf = []
    # insert each row
    for i in range(len(df)):
        # insert the new categoritcal distance values in the dataframe
        newRow = list(df.iloc[i].values) + one_hot[i] + [label[i]]
        newDf.append(newRow)

    # get column names
    for i in range(len(one_hot[0])):
        col_names.append("one_hot_" + str(i+1))
    columns = list(df.columns) + col_names + ['label']

    # return a new dataframe
    return pd.DataFrame(newDf, columns=columns)

if __name__ == "__main__":
    k = [1,2,]
    # get the data
    df = pd.read_csv(DATA_PATH)
    df = make_categorical(df)
    
    # -----------------------------to drop or not to drop distance column--------------------
    # df = df.drop(['distance'], axis=1)
    #---------------------------------------------------------------------------------------

    # get the list of sequences
    seq = get_seq(df)

    #add columns for the kmers
    for num in k:
        print("now doing kmer number", num)
        kmer_dict, comb_output = get_col(num, seq, df['bpos1'].values, df['bpos2'].values)
        df = add_col(df, num, kmer_dict, comb_output)

    # # export to a new file
    df.to_csv(NEW_DATA_PATH)
