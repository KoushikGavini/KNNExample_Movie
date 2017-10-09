import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix

# df = pd.read_fwf('train.dat', header=None, 
#         widths=[2, int(1e10)], names=['Sentiment', 'Review'])

# print(df)

import csv

with open('/Users/wppa/Desktop/CMPE_139_Knn/KNNExample_Movie/train.dat','r') as f:
    df = pd.DataFrame(l.rstrip().split() for l in f)

# print(df)
vals = df.ix[:,:].values
# print(vals)
Sentiments_names = [n[0][0:] for   n in vals]
# print(Sentiments_names)
names = [n[2][5:] for n in vals]
# print(names)

# cls = [n[0][0] for n in vals]

def cmer(name, c=3):
    r""" Given a name and parameter c, return the vector of c-mers associated with the name
    """
    name = name.lower()
    if len(name) < c:
        return [name]
    v = []
    for i in range(len(name)-c+1):
        v.append(name[i:(i+c)])
    return v
def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat

def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat
        
def namesToMatrix(names, c):
    docs = [cmer(n, c) for n in names]
    return build_matrix(docs)
# From Activity nn-2-Sol
#We'll now define a function to search for the top- kk  neighbors for a given name (one of the objects the dataset), where proximity is computed via cosine similarity.
def findNeighborsForName(name, c=1, k=1):
    # first, find the document for the given name
    id = -1
    for i in range(len(names)):
        if names[i] == name:
            id = i
            break
    if id == -1:
        print("Name %s not found." % name)
        return []
    # now, compute similarities of name's vector against all other name vectors
    mat = namesToMatrix(names, c)
    csr_l2normalize(mat)
    x = mat[id,:]
    dots = x.dot(mat.T)
    dots[0,id] = -1 # invalidate self-similarity
    sims = list(zip(dots.indices, dots.data))
    sims.sort(key=lambda x: x[1], reverse=True)
    return [names[s[0]] for s in sims[:k] if s[1] > 0 ]
findNeighborsForName("ing", c=2, k=5)