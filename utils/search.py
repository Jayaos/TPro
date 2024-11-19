import numpy as np
import structures.transcription_factors as tfs
from constructor.tp_graph import TPGraph, tp_graph_constructor_sepa

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


def graph_from_matrix(this_representation):
    cassettes = []
    for i in range(this_representation.shape[0]):
        
        # Layer 1 (constitutive TFs)
        if this_representation[i,0]:
            constit_tf = tfs.TranscriptionFactor(tf_lookup[i][0], tf_lookup[i][1], tf_lookup[i][2])
            cassettes.append(tfs.Cassette("constitutive", constit_tf))
        
        # Layer 2 Channel 1
        if this_representation[i,1]:
            layer_2_tf = tfs.TranscriptionFactor(tf_lookup[i][0], tf_lookup[i][1], tf_lookup[i][2])
            cassettes.append(tfs.Cassette("1", layer_2_tf))

        # Layer 2 Channel 2
        if this_representation[i,2]:
            layer_2_tf = tfs.TranscriptionFactor(tf_lookup[i][0], tf_lookup[i][1], tf_lookup[i][2])
            cassettes.append(tfs.Cassette("2", layer_2_tf))

        # Layer 2 Channel 3
        if this_representation[i,3]:
            layer_2_tf = tfs.TranscriptionFactor(tf_lookup[i][0], tf_lookup[i][1], tf_lookup[i][2])
            cassettes.append(tfs.Cassette("3", layer_2_tf))

        # Layer 2 Channel 4
        if this_representation[i,4]:
            layer_2_tf = tfs.TranscriptionFactor(tf_lookup[i][0], tf_lookup[i][1], tf_lookup[i][2])
            cassettes.append(tfs.Cassette("4", layer_2_tf))

    # Layer 3 (directly regulate output)
    cassettes.append(tfs.Cassette("5", tfs.TranscriptionFactor("O", "", "")))

    tp_graph = TPGraph(input_ligands=rcds, init_root=False)
    tp_graph.root, n_roots = tp_graph_constructor_sepa(cassettes,verbose=False)
    
    return tp_graph, n_roots

def flip_expression(tf_tup):
    if tf_tup[-1] == "+":
        new_tups = [(tf_tup[0], tf_tup[1], "A"), (tf_tup[0], tf_tup[1], "S")]
    elif tf_tup[-1] == "A":
        new_tups = [(tf_tup[0], tf_tup[1], "+"), (tf_tup[0], tf_tup[1], "S")]
    elif tf_tup[-1] == "S":
        new_tups = [(tf_tup[0], tf_tup[1], "+"), (tf_tup[0], tf_tup[1], "A")]
    return new_tups

def flip_expression_list(tf_tup_list):
    if isinstance(tf_tup_list[0],tuple):
        flipped_list = []
        for tf_tup in tf_tup_list:
            if tf_tup[-1] == "+":
                flipped_list += [(tf_tup[0], tf_tup[1], "A"), (tf_tup[0], tf_tup[1], "S")]
            elif tf_tup[-1] == "A":
                flipped_list += [(tf_tup[0], tf_tup[1], "+"), (tf_tup[0], tf_tup[1], "S")]
            elif tf_tup[-1] == "S":
                flipped_list += [(tf_tup[0], tf_tup[1], "+"), (tf_tup[0], tf_tup[1], "A")]
    else:
        if tf_tup_list[-1] == "+":
            flipped_list = [(tf_tup_list[0], tf_tup_list[1], "A"), (tf_tup_list[0], tf_tup_list[1], "S")]
        elif tf_tup_list[-1] == "A":
            flipped_list = [(tf_tup_list[0], tf_tup_list[1], "+"), (tf_tup_list[0], tf_tup_list[1], "S")]
        elif tf_tup_list[-1] == "S":
            flipped_list = [(tf_tup_list[0], tf_tup_list[1], "+"), (tf_tup_list[0], tf_tup_list[1], "A")]
    return flipped_list

def check_validity(this_representation, verbose=False):
    '''
    Make sure a circuit is functional
    '''

    # Rule 0 : promoter cannot regulate tf with same dbd as promoter
    for j in range(1,this_representation.shape[1]):
        this_dbd = column_to_dbd_map[j]
        this_dbd_idx = dbd_idx[this_dbd]
        if this_representation[this_dbd_idx,j].sum() != 0:
            if verbose:
                print("Violated Rule 0")
            return False

    for i in range(this_representation.shape[0]):

        # Rule 1 : only one phenotype per constitutively expressed transcription factor
        flipped_1, flipped_2 = flip_expression(tf_lookup[i])
        if this_representation[i,0] and (this_representation[tf_idx_lookup[flipped_1],0] or this_representation[tf_idx_lookup[flipped_2],0]):
            if verbose:
                print("Violated Rule 1")
            return False


        # Rule 2 : identical TFs cannot be expressed by two different promoters
        if this_representation[i,1:].sum() > 1:
            if verbose:
                print("Violated Rule 2")
            return False


        # Rule 3 : constit rcd cannot regulate same rcd on layer 2
        if this_representation[i,0]:
            constit_dbd, constit_rcd, _ = tf_lookup[i]
            if constit_dbd != '5':
                this_col = dbd_to_column_map[constit_dbd]
                for j in range(this_representation.shape[0]):
                    if this_representation[j,this_col] and tf_lookup[j][1] == constit_rcd:
                        if verbose:
                            print("Violated Rule 3 (RCD {} regulates TF {})".format(constit_rcd, tf_lookup[j]))
                        return False

    # Rule 4 : at least one TF with promoter 5 must be present
    if this_representation[dbd_idx["5"]].sum() == 0:
        if verbose:
            print("Violated Rule 4")
        return False

    return True

def check_single_root(this_representation):
    
    # Rule 4 : at least one TF with promoter 5 must be present
    if not this_representation[dbd_col == '5'].sum():
        return False
    
    for i in range(1,5):
        this_dbd_row_idx = np.where(dbd_col == column_to_dbd_map[i])[0]
        
        expressed_i = this_representation[this_dbd_row_idx,:].any()
        promoter_i = this_representation[:,i].any()
        
        # if any TFs with dbd_i are expressed, make sure the promoter at dbd_i expresses something
        if expressed_i and not promoter_i:
            return False
        if promoter_i and not expressed_i:
            return False
        
    _, n_roots = graph_from_matrix(this_representation)
    return n_roots == 1

def get_next_valid_idx(this_representation, num_constit=1, num_channel=3, parallel=False):
    '''
    Given a partial representation (sparse binary matrix), return a collection of indices.
    Each index corresponds to adding a component which does not break the circuit.
    '''
    
    # boolean mask which says where we are able to put the next item
    valid_mask = np.full(this_representation.shape,False)
    
    # get a list of dbds and rcds (with corresponding indices) which are consitutively expressed
    constit_idx = np.where(this_representation[:,0])[0]
    constit_dbds = dbd_col[constit_idx]
    constit_rcds = rcd_col[constit_idx]

    # these are the indices of the columns at which constitutive TFs can attach
    valid_cols = np.array([dbd_to_column_map[_dbd] for _dbd in constit_dbds])

    # a mask of all TFs which are not yet expressed (for Rule 2)
    valid_promoter_row_mask = ~this_representation.any(axis=1)
    
    # identify the TFs (rcds and source dbds) which are input to each node (dbd)
    all_expressed_tfs_mask = this_representation.sum(axis=1).astype(bool)
    node_rcds, node_dbds = {}, {}
    for i in [1,2,3,4]:

        # all expressed TFs with this dbd
        node_input_tfs = np.logical_and(all_expressed_tfs_mask, dbd_col == column_to_dbd_map[i])

        # list of rcds corresponding to TFs which are input to this node (dbd)
        node_rcds[i] = rcd_col[node_input_tfs]

        # list of columns (nodes/dbds) which regulate a TF that is input to this node (dbd)
        this_node_dbds = np.where(this_representation[node_input_tfs].sum(axis=0))[0]
        this_node_dbds = this_node_dbds[this_node_dbds != 0] # dont worry about constit here
        this_node_dbds = [column_to_dbd_map[_this_dbd] for _this_dbd in this_node_dbds]
        node_dbds[i] = np.array(this_node_dbds)


    # apply rules to prune search space


    # apply rules to constitutive
    valid_constit_mask = np.full((this_representation.shape[0]),True)
    if this_representation[:,0].sum() > 0:
        flipped_exp_idx = np.array(itemgetter(*flip_expression_list(itemgetter(*np.where(this_representation[:,0])[0])(tf_lookup)))(tf_idx_lookup))

        # Rule 1 : only one phenotype per constitutively expressed transcription factor
        valid_constit_mask[flipped_exp_idx] = False
    
    # Rule 2 : identical TFs cannot be expressed by two different promoters
    valid_constit_mask = np.logical_and(valid_constit_mask, valid_promoter_row_mask)
    
    # Rule 3 : constit rcd cannot regulate same rcd on layer 2
    for i in range(1,5):
        regulated_rcd = rcd_col[np.where(this_representation[:,i])[0]]
        if len(regulated_rcd) != 0:
            valid_constit_mask = np.logical_and(valid_constit_mask,
                                                ~np.logical_and(dbd_col == column_to_dbd_map[i],
                                                                rcd_col == regulated_rcd[0]))

    # Rule 5 : don't add item if it is already present
    valid_constit_mask = np.logical_and(valid_constit_mask, ~(this_representation[:,0].astype(bool)))
    

    # Temporary Rule - allow at most num_constit constit
    if this_representation[:,0].sum() >= num_constit:
        valid_constit_mask = np.full((this_representation.shape[0]),False)
        
    valid_mask[:,0] = valid_constit_mask


    # apply rules to promoters
    for i in range(1,5):
        
        # Temporary Rule - allow at most num_channel channels
        if i in list(range(num_channel+1,5)):
            continue
            
        # Rule 3 : constit rcd cannot regulate same rcd on layer 2
        this_col_idx = np.where(valid_cols == i)[0]
        if len(this_col_idx) > 0:
            valid_mask[:,i] = (rcd_col[...,None] != constit_rcds[None,this_col_idx]).all(axis=1)
        else:
            valid_mask[:,i] = np.full(valid_mask[:,i].shape,True)
            
        # Rules for parallel promoters
        if len(node_dbds[i]) > 0:
            # Rule 6 : TF cannot regulate same dbd from which it is regulated (no "loops")
            valid_mask[:,i] = np.logical_and(valid_mask[:,i],
                                             (dbd_col[...,None] != node_dbds[i][None]).all(axis=1))
            
            # Rule 3 : rcd cannot regulate same rcd
            valid_mask[:,i] = np.logical_and(valid_mask[:,i],
                                             (rcd_col[...,None] != node_rcds[i][None]).all(axis=1))

        # Rule 0 : promoter cannot regulate tf with same dbd as promoter
        valid_mask[:,i] = np.logical_and(valid_mask[:,i],
                                         dbd_col != column_to_dbd_map[i])

        # Rule 2 : identical TFs cannot be expressed by two different promoters
        valid_mask[:,i] = np.logical_and(valid_mask[:,i],
                                         valid_promoter_row_mask)

        if parallel:
            # Temporary Rule - only allow parallel channels
            valid_mask[:,i] = np.logical_and(valid_mask[:,i],
                                             dbd_col == "5")

        # Rule 5 : don't add item if it is already present
        valid_mask[:,i] = np.logical_and(valid_mask[:,i],
                                         ~this_representation[:,i].astype(bool).any().repeat(this_representation.shape[0]))


    # indices of representation components which are able to be added
    valid_idx = np.concatenate([_idx.reshape(-1,1) for _idx in np.where(valid_mask)],axis=-1)
    
    return valid_idx[np.argsort(valid_idx[:,1])][::-1]

def search_circuits_recursive(this_representation, checked_circuits, logic_table_dict, num_constit=1, num_channel=3, parallel=False):

    checked_circuits.append(hex(int("".join(list(this_representation.flatten().astype(str))), 2)))

    # indices of representation components which are able to be added
    valid_idx = get_next_valid_idx(this_representation, num_constit=num_constit, num_channel=num_channel, parallel=parallel)
    
    for idx_i, idx_j in valid_idx:
        next_representation = np.copy(this_representation)

        next_representation[idx_i, idx_j] = 1

        if hex(int("".join(list(next_representation.flatten().astype(str))), 2)) in checked_circuits:
            continue

        # check if we have a valid circuit already (only one root)
        # temporary rule - exactly num_channel channels and exactly num_constit constit
        if next_representation[:,1:].sum() >= num_channel and \
        next_representation[:,0].sum() >= num_constit and \
        check_single_root(next_representation):
            
            assert check_validity(next_representation,verbose=True), 'missed a rule!'
            
            tp_graph, n_roots = graph_from_matrix(next_representation)

            assert n_roots == 1, 'check_single_root missed a root!'

            logic_str = ""
            for inp in three_inp_eval:
                logic_str += str(tp_graph.evaluate(inp))
            if logic_str not in logic_table_dict:
                logic_table_dict[logic_str] = [scipy.sparse.csr_matrix(next_representation)]
                print("Found {} Logic Tables".format(len(logic_table_dict.keys())))
            else:
                logic_table_dict[logic_str].append(scipy.sparse.csr_matrix(next_representation))

        # if not yet valid (multiple roots), keep adding items
        else:
            search_circuits_recursive(next_representation,
                                      checked_circuits,
                                      logic_table_dict,
                                      num_constit=num_constit,
                                      num_channel=num_channel,
                                      parallel=parallel)
            
def update_merged_circuits_dict(all_keys, merged_smallest_circuits, smallest_circuits_thisconfig):
    
    merged_nnz = []
    
    for key in tqdm(all_keys):

        possible_circuits = []
        possible_circuits_nnz = []

        if key in merged_smallest_circuits:
            
            for circuit in merged_smallest_circuits[key]:
            
                possible_circuits.append(circuit)
                possible_circuits_nnz.append(circuit.nnz)

        if key in smallest_circuits_thisconfig:
            
            for circuit in smallest_circuits_thisconfig[key]:
            
                possible_circuits.append(circuit)
                possible_circuits_nnz.append(circuit.nnz)

        min_nnz = np.min(possible_circuits_nnz)
        min_nnz_idx = (np.array(possible_circuits_nnz) == min_nnz).nonzero()[0]
        merged_smallest_circuits[key] = np.array(possible_circuits)[min_nnz_idx]
        merged_nnz.append(min_nnz)
        
    return merged_smallest_circuits, merged_nnz

def iterate_with_conditions(condition_list: list):
    
    """
    look up copressed circuits in recursive way given the condition list
    --args
    condition_list: list of conditions with the number of constitutive layer(s) and channel(s)
    """
    this_parallel = False
    rep_height = 45 # total number of possible TFs
    rep_width = 1 + 4 # constitutive TF + (4 * layer_2_TF)
    all_keys = set()
    merged_smallest_circuits = {}
    
    for cond in condition_list:
        
        this_num_constit, this_num_channel = cond
        checked_circuits_thisconfig = []
        logic_table_dict_thisconfig = {}
        this_representation = np.zeros((rep_height, rep_width),dtype=int)
        
        print("search circuit space with {} constitutive and {} channels".format(this_num_constit, this_num_channel))
        search_circuits_recursive(this_representation, checked_circuits_thisconfig, logic_table_dict_thisconfig, 
                                  num_constit=this_num_constit, num_channel=this_num_channel, parallel=this_parallel)
        
        print("updating the search results...")
        smallest_circuits_thisconfig = {}
        for key in tqdm(logic_table_dict_thisconfig):
            nnz_list = []
            
            for i in range(len(logic_table_dict_thisconfig[key])):
                nnz_list.append(logic_table_dict_thisconfig[key][i].nnz)
                min_nnz = np.min(nnz_list)
                min_nnz_idx = (np.array(nnz_list) == min_nnz).nonzero()[0]
        
            smallest_circuits_thisconfig[key] = np.array(logic_table_dict_thisconfig[key])[min_nnz_idx]
        
        print("merging the search results...")
        all_keys = set(list(all_keys) + list(smallest_circuits_thisconfig.keys()))
        merged_smallest_circuits, merged_nnz = update_merged_circuits_dict(all_keys, merged_smallest_circuits, 
                                                                           smallest_circuits_thisconfig)
    
    return merged_smallest_circuits, merged_nnz