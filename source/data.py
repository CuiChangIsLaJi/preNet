from Bio import Seq,SeqRecord,SeqIO
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import RNA
import pandas as pd
import numpy as np
import re

class preNet_dataset(Dataset):
    def __init__(self,seq_features,struct_features,labels,scaled=False):
        self.seq_features = seq_features
        self.struct_features = struct_features
        labels = np.array(labels).reshape(-1,1)
        if not scaled:
            labels = StandardScaler(with_std=False).fit_transform(labels)
        self.labels = torch.tensor([[1] if label < 0 else [0] for label in labels]).double()
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,idx):
        return self.seq_features[idx],self.struct_features[idx],self.labels[idx]

def parse_and_split_samples(seq_file,struct_file,score_file,n_samples,split=True,test_size=0.1):
    seq_record = SeqIO.parse(seq_file,"fasta")
    if struct_file=="":
        struct_record = predict_2D(seq_record)
    else:
        struct_record = SeqIO.parse(struct_file,"fasta")

    seqs = [str(seq.seq) for seq in seq_record]
    structs = [str(struct.seq) for struct in struct_record]
    scores = pd.read_csv(score_file,sep="\t")
    
    if split:
        train_idx, test_idx = train_test_split(range(n_samples),test_size=test_size)
        seqs_train = [sequence_encoder(seqs[i]) for i in train_idx]
        seqs_test = [sequence_encoder(seqs[j]) for j in test_idx]
        structs_train = [struct_encoder(structs[k]) for k in train_idx]
        structs_test = [struct_encoder(structs[l]) for l in test_idx]
        scores_train = scores["average_score"][train_idx]
        scores_test = scores["average_score"][test_idx]
        return seqs_train,seqs_test,structs_train,structs_test,scores_train,scores_test
    else:
        seqs = [sequence_encoder(seq) for seq in seqs]
        structs = [struct_encoder(struct) for struct in structs]
        scores = scores["average_score"]
        return seqs,structs,scores

def predict_2D(seqs):
    return [SeqRecord.SeqRecord(Seq.Seq(RNA.fold(str(seq.seq))[0]),id=seq.id) for seq in seqs]

base={"A":0, "C":1, "G":2, "U":3}
element={"(":0, ")":1, ".":2, ":":3}

def loop_conversion(struct):
    '''
    params
    -------------------------------------
    struct(str): The secondary structure string in dot-bracket format, one of "(",")",".",":". The unpaired residue inside a loop is determined via either the colon mark(colonized=True) or the automatic conversion LoopConversion(colonized=False).
    -------------------------------------
    returns
    -------------------------------------
    A new string with loop elements in new format ":".
    -------------------------------------
    reference
    -------------------------------------
    The loop conversion scheme is an integration and re-implementation of a work by Seunghyun Park et al. See https://github.com/eleventh83/deepMiRGene/blob/master/inference/deepMiRGene.py for the original code implementation.
    '''
    new_struct = struct[:]
    pattern = re.compile("\(+(\.)+\)+")
    for loop in pattern.finditer(struct):
        new_struct = "".join((new_struct[:loop.regs[1][0]],":"*(loop.regs[1][1]-loop.regs[1][0]),new_struct[(loop.regs[1][1]):]))
    return new_struct

def sequence_encoder(seq):
    '''
    params
    -------------------------------------
    seq(str): The sequence in fasta format, composed of "A","G","C","U".
    -------------------------------------
    returns
    -------------------------------------
    A one-hot-encoding-based (L,4) DataFrame, where L is the sequence length.
    '''
    L = len(seq)
    feature = np.zeros((L,4))
    for row in range(L):
        col = base[seq[row]]
        feature[row,col]=1
    return torch.from_numpy(feature)

def struct_encoder(struct,colonized=False):
    '''
    params
    -------------------------------------
    struct(str): The secondary structure string in dot-bracket format, composed of "(",")",".",":". The unpaired residue inside a loop is determined via either the colon mark(colonized=True) or the automatic conversion LoopConversion(colonized=False).
    -------------------------------------
    returns
    -------------------------------------
    A one-hot-encoding-based (L,4) DataFrame, where L is the sequence length.
    -------------------------------------
    '''
    L = len(struct)
    feature = np.zeros((L,4))
    if not colonized:
        struct = loop_conversion(struct)
    for row in range(L):
        col = element[struct[row]]
        feature[row,col]=1
    return torch.from_numpy(feature)
