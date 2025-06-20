import numpy as np


# env variables
dbds = ['1','2','3','4','5']
rcds = ['A', 'B', 'C']
exps = ["+", "A", "S"]
three_inp_eval = [[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,1,1]]

dbd_to_column_map = {
    "1" : 1,
    "2" : 2,
    "3" : 3,
    "4" : 4,
    "5" : None,
}

column_to_dbd_map = {v:k for k, v in dbd_to_column_map.items()}

tf_lookup = {}
counter = 0
for dbd in dbds:
    for rcd in rcds:
        for exp in exps:
            tf_lookup[counter] = (dbd, rcd, exp)
            counter += 1
tf_idx_lookup = {v:k for k, v in tf_lookup.items()}


dbd_idx = {}
for dbd in dbds:
    this_dbd_idx = []
    for rcd in rcds:
        for exp in exps:
            this_dbd_idx.append(tf_idx_lookup[(dbd,rcd,exp)])
    dbd_idx[dbd] = np.array(this_dbd_idx)

    dbd_col, rcd_col, exp_col = [], [], []
    
for i in range(len(tf_lookup.keys())):
    dbd_col.append(tf_lookup[i][0])
    rcd_col.append(tf_lookup[i][1])
    exp_col.append(tf_lookup[i][2])
    
dbd_col = np.array(dbd_col)
rcd_col = np.array(rcd_col)
exp_col = np.array(exp_col)