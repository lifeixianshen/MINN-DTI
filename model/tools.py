import numpy as np
import re
import torch
from torch.autograd import Variable
import time
from chemprop.data.utils import get_data_from_smiles
from chemprop.features import mol2graph
def get_torch_device():
    """
    Getter for an available pyTorch device.
    :return: CUDA-capable GPU if available, CPU otherwise
    """
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
def create_variable(tens):
    # Do cuda() before wrapping with variable
    return Variable(torch.tensor(tens).to(get_torch_device()))
def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string
# Create necessary variables, lengths, and target

def make_variables(lines, properties,letters):
    sequence_and_length = [line2voc_arr(line,letters) for line in lines]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences(vectorized_seqs, seq_lengths, properties)

def make_variables_s(lines,properties,mpnargs):
    return  mol2graph(lines,mpnargs),create_variable(properties)


def make_variables_seq(lines,letters):
    sequence_and_length = [line2voc_arr(line,letters) for line in lines]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences_seq(vectorized_seqs, seq_lengths)
def line2voc_arr(line,letters):
    arr = []
    regex = '(\[[^\[\]]{1,10}\])'
    line = replace_halogen(line)
    char_list = re.split(regex, line)
    for char in char_list:
        if char.startswith('['):
            arr.append(letterToIndex(char,letters))
        else:
            chars = list(char)

            arr.extend(letterToIndex(unit,letters) for unit in chars)
    return arr, len(arr)
def letterToIndex(letter,smiles_letters):
    return smiles_letters.index(letter)
# pad sequences and sort the tensor
def pad_sequences(vectorized_seqs, seq_lengths, properties):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # Also sort the target (countries) in the same order
    target = properties.double()
    if len(properties):
        target = target[perm_idx]
    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor),create_variable(seq_lengths),create_variable(target)
def pad_sequences_seq(vectorized_seqs, seq_lengths):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor), create_variable(seq_lengths)

def construct_vocabulary(smiles_list,fname):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    regex = '(\[[^\[\]]{1,10}\])'
    for smiles in smiles_list:
        smiles = ds.replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = list(char)
                [add_chars.add(unit) for unit in chars]

    print(f"Number of characters: {len(add_chars)}")
    with open(fname, 'w') as f:
        f.write('<pad>' + "\n")
        for char in add_chars:
            f.write(char + "\n")
    return add_chars
def readLinesStrip(lines):
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')
    return lines
def getProteinSeq(path,contactMapName):
    proteins = open(f"{path}/{contactMapName}").readlines()
    proteins = readLinesStrip(proteins)
    return proteins[1]
def getProtein(path,contactMapName,contactMap = True):
    proteins = open(f"{path}/{contactMapName}").readlines()
    proteins = readLinesStrip(proteins)
    seq = proteins[1]
    if contactMap:
        contactMap = [proteins[i] for i in range(2,len(proteins))]
        return seq,contactMap
    else:
        return seq

def getTrainDataSet(trainFoldPath):
    with open(trainFoldPath, 'r') as f:
        trainCpi_list = f.read().strip().split('\n')
    return [cpi.strip().split() for cpi in trainCpi_list]
def getTestProteinList(testFoldPath):
    return readLinesStrip(open(testFoldPath).readlines())[0].split()
def getSeqContactDict(contactPath,contactDictPath):# make a seq-contactMap dict 
    contactDict = open(contactDictPath).readlines()
    seqContactDict = {}
    for data in contactDict:
        seq,contactMapName = data.strip().split(':')
        _,contactMap = getProtein(contactPath,contactMapName)
        contactmap_np = [list(map(float, x.strip(' ').split(' '))) for x in contactMap]
        feature2D = np.expand_dims(contactmap_np, axis=0)
        feature2D = torch.FloatTensor(feature2D)    
        seqContactDict[seq] = feature2D
    return seqContactDict
def getLetters(path):
    with open(path, 'r') as f:
        chars = f.read().split()
    return chars
def getDataDict(testProteinList,activePath,decoyPath,contactPath):
    dataDict = {}
    for x in testProteinList:
        protein = x.split('_')[0]
        proteinActPath = f"{activePath}/{protein}_actives_final.ism"
        proteinDecPath = f"{decoyPath}/{protein}_decoys_final.ism"
        act = open(proteinActPath,'r').readlines()
        dec = open(proteinDecPath,'r').readlines()
        actives = [[x.split(' ')[0],1] for x in act] ######
        decoys = [[x.split(' ')[0],0] for x in dec]# test
        seq = getProtein(contactPath,x,contactMap = False)
        xData = [[active[0], seq, active[1]] for active in actives]
        xData.extend([decoy[0], seq, decoy[1]] for decoy in decoys)
        dataDict[x] = xData
    return dataDict

def my_collate(batch):
    smiles = [item[0] for item in batch]
    contactMap = [item[1] for item in batch]
    label = [item[2] for item in batch]
    seq = [item[3] for item in batch]
    return [smiles, contactMap, label,seq]

def time_log(s):
    print(f'{time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}-{s}')
    return