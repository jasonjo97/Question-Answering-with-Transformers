import glob  
import numpy as np 
import pandas as pd   


def findFiles(path): return glob.glob(path)

def train_test_split(data): 
    """ Split into training and test dataset """
    split_ratio = 0.1 
    random_seed = 10 
    
    indices = list(range(len(data)))
    split = int(np.floor(split_ratio * len(data)))

    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, test_idx = indices[split:], indices[:split]

    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]
    
    return train_data, test_data


def GenerateProof(data): 
    """ Return a list of proofs where proof is defined as a sequence of two entities and their relationship """ 
    proof_list = [] 

    for row in range(data["genders"].size):
        names = list(map(lambda x: x.split(":")[0], data["genders"][row].split(","))) 

        proof = []    

        for idx in range(len(eval(data["story_edges"][row]))):
            edges = eval(data["story_edges"][row])[idx]
            proof.append([names[edges[0]], eval(data["edge_types"][row])[idx], names[edges[1]]])

        proof_list.append(proof)
        
    return proof_list


def GeneratePairs(data): 
    """ Return a list of pairs where a pair consists of a story and it's following proofs """
    pairs = []
    proof = GenerateProof(data)
    
    for row in range(data["story"].size):
        # add "SEP" token at the end of each proof sequence 
        n = len(proof[row])
        proof_eos = ''
        for i in range(n):
            proof_eos += ' '+' '.join(proof[row][i])+' SEP'
        pairs.append([data["story"][row], proof_eos[1:]])
    
    return pairs 


def NumofEntities(data): 
    """ Return two lists 
    @return   name_list (list): List of all possible entity names 
    @return   num_list (list): List of first N positive integers, where N is the number of all possible entities 
    """ 
    name_list = []
    for row in range(data["query"].size): 
        name_list.append(eval(data["query"][row])[0])
        name_list.append(eval(data["query"][row])[1])

    name_list = list(set(name_list))

    for filename in findFiles('./datasets/1.*_test.csv'):
        test = pd.read_csv(filename)

        for row in range(test["query"].size): 
            name_list.append(eval(test["query"][row])[0])
            name_list.append(eval(test["query"][row])[1])

        name_list = list(set(name_list))

    num_list = [i for i in range(len(name_list))]
    
    return name_list, num_list 


def NumofRelationships(data):
    """ Return a list of all possible relationships """
    rel_list = []

    for row in range(data["edge_types"].size): 
        rel_list.append(eval(data["edge_types"][row])[0])
        rel_list.append(eval(data["edge_types"][row])[1])

    rel_list = list(set(rel_list))

    for filename in findFiles('./datasets/1.*_test.csv'):
        test = pd.read_csv(filename)

        for row in range(test["query"].size): 
            rel_list.append(eval(test["edge_types"][row])[0])
            rel_list.append(eval(test["edge_types"][row])[1])

        rel_list = list(set(rel_list))
    
    return rel_list 