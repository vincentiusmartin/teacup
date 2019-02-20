# This script adds kmer features to the existing data

import pandas as pd


NEW_DATA_PATH = "../data/generated/training_kmer.csv"
DATA_PATH = "../data/generated/training.csv"
def get_seq(data):
	'''
	This function returns the list of sequences from the data
	in a numpy array
	Input: input data as a dataframe
	Output: DNA sequences in the data in a numpy array
	'''

	return data["sequence"].values

def get_kmer(num, seq):
	'''
	Number of distinct k-mer in the sequence
	Input: num, seq where num = k and seq is the sequence
		   to be evaluated
	Output: len(kmer_set) which is the number of distinct k-mers
		    in the sequence
	'''

	# initialize an empty set
	kmer_set = set()
	# initialize index i to 0
	i = 0
	# iterate through the sequence
	while i + num < len(seq):
		# get the read of length num
		read = seq[i:i+num]
		# add this read to the set
		kmer_set.add(read)
		# increment index
		i += 1

	# return number of elements in the set
	return len(kmer_set)

def get_col(num, seq_col):
	# initialize an empty list to store the output
	output = []
	
	# iterate through each sequence in the data
	for seq in seq_col:
		# append the number of distinct k-mer in the sequence
		# to the output list
		output.append(get_kmer(num, seq))

	# return the output
	return output

def add_col(df, num, col):
	'''
	This function inserts the new column into the dataframe
	Input: 
	Output: df which is the updated dataframe
	'''

	# set the new column's name
	col_name = "num_" + str(num) +"mer"

	# insert the new column's name in the dataframe
	df.insert(df.shape[1] - 1, col_name, col)

	# return the updated dataframe
	return df

if __name__ == "__main__":
	k = [1,2,3]
	# get the data
	df = pd.read_csv(DATA_PATH)
	# get the list of sequences
	seq = get_seq(df)
	# add columns for the kmers
	for num in k:
		print("now doing kmer number", num)
		new_col = get_col(num, seq)
		df = add_col(df, num, new_col)

	# export to a new file
	df.to_csv(NEW_DATA_PATH)
