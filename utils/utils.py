import os
import pickle
import numpy as np
from tqdm import tqdm
from constructor.tp_graph import TPGraph
from structures.nodes import *
from structures.transcriptional_programs import *
from structures.transcription_factors import *
from variables.variables import *


def load_data(data_dir):
    file = open(data_dir,'rb')
    
    return pickle.load(file)


def save_data(save_dir, data_dict):
    with open(save_dir, 'wb') as f:
        pickle.dump(data_dict, f)


def safe_save_pickle(filename, obj):
    temp_filename = filename + ".tmp"

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    try:
        with open(temp_filename, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())  # ensure all data is written to disk
        os.replace(temp_filename, filename)  # atomic replace
    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)



def tp_graph_constructor_sepa(cassettes, verbose=False):

    tp_nodes = []
    for cassette in cassettes:
        if cassette.promoter == "constitutive":
            tp_nodes.append(ConstitutiveNode(cassette.transcription_factor))
            if verbose:
                print(tp_nodes[-1])
        else:
            incident_tfs = []
            for _cassette in cassettes:
                if (_cassette.transcription_factor.dbd == cassette.promoter) and (_cassette.transcription_factor not in incident_tfs):
                    incident_tfs.append(_cassette.transcription_factor)
            tp_nodes.append(TPNode(program=SEPAProgram(incident_tfs,output=cassette.transcription_factor), parents=[]))
            if verbose:
                print(tp_nodes[-1])
                    
    
    # identify the parents of each node
    # parents produce TFs which are used as inputs to a node's program
    for i in range(len(tp_nodes)):
        if isinstance(tp_nodes[i], ConstitutiveNode):
            continue
        for j in range(len(tp_nodes)):
            if isinstance(tp_nodes[j], ConstitutiveNode):
                out_j = tp_nodes[j].output
            else:
                out_j = tp_nodes[j].program.output
            if out_j in tp_nodes[i].program.transcription_factors:
                tp_nodes[i].parents.append(tp_nodes[j])


    # find the node which has no children
    n_children_dict = {}
    n_roots = 0
    for node in tp_nodes:
        n_children_dict[node] = 0
        for _node in tp_nodes:
            if (not isinstance(_node, ConstitutiveNode)) and node in _node.parents:
                n_children_dict[node] += 1
        if n_children_dict[node] == 0:
            n_roots += 1
            if verbose:
                print('root found')
            root_node = node

    return root_node, n_roots


def update_merged_circuits_dict(all_keys, merged_smallest_circuits, smallest_circuits_thisconfig, merged_circuit_desc, this_circuit_desc):
    merged_nnz = []
    for key in tqdm(all_keys):

        possible_circuits = []
        possible_circuits_nnz = []
        possible_circutis_desc = []

        if key in merged_smallest_circuits:
            possible_circuits.append(merged_smallest_circuits[key])
            possible_circuits_nnz.append(merged_smallest_circuits[key].nnz)
            possible_circutis_desc.append(merged_circuit_desc[key])

        if key in smallest_circuits_thisconfig:
            possible_circuits.append(smallest_circuits_thisconfig[key])
            possible_circuits_nnz.append(smallest_circuits_thisconfig[key].nnz)
            possible_circutis_desc.append(this_circuit_desc)

        merged_smallest_circuits[key] = possible_circuits[np.argmin(possible_circuits_nnz)]
        merged_circuit_desc[key] = possible_circutis_desc[np.argmin(possible_circuits_nnz)]
        merged_nnz.append(np.min(possible_circuits_nnz))
        
    return merged_smallest_circuits, merged_circuit_desc, merged_nnz


def merge_circuit_search_results(data_dir):
    
    file_list = os.listdir(data_dir)
    all_keys = set()
    merged_smallest_circuits = {}
    merged_circuit_desc = {}
    
    for f in file_list:
        print("merging {}".format(f))
    
        circuit_search_result = load_data(data_dir + f)
        all_keys = all_keys.union(set(list(circuit_search_result.keys())))
        smallest_circuits, circuit_desc, merged_nnz = update_merged_circuits_dict(all_keys, merged_smallest_circuits, circuit_search_result, merged_circuit_desc, f)
        merged_circuit_desc.update(circuit_desc)
        merged_smallest_circuits.update(smallest_circuits)
        
    return merged_smallest_circuits, merged_circuit_desc


def graph_from_matrix(this_representation):
    cassettes = []
    for i in range(this_representation.shape[0]):
        
        # Layer 1
        if this_representation[i,0]:
            constit_tf = TranscriptionFactor(tf_lookup[i][0], tf_lookup[i][1], tf_lookup[i][2])
            cassettes.append(Cassette("constitutive", constit_tf))
        
        # Layer 2 Channel 1
        if this_representation[i,1]:
            layer_2_tf = TranscriptionFactor(tf_lookup[i][0], tf_lookup[i][1], tf_lookup[i][2])
            cassettes.append(Cassette("1", layer_2_tf))

        # Layer 2 Channel 2
        if this_representation[i,2]:
            layer_2_tf = TranscriptionFactor(tf_lookup[i][0], tf_lookup[i][1], tf_lookup[i][2])
            cassettes.append(Cassette("2", layer_2_tf))

        # Layer 2 Channel 3
        if this_representation[i,3]:
            layer_2_tf = TranscriptionFactor(tf_lookup[i][0], tf_lookup[i][1], tf_lookup[i][2])
            cassettes.append(Cassette("3", layer_2_tf))

        # Layer 2 Channel 4
        if this_representation[i,4]:
            layer_2_tf = TranscriptionFactor(tf_lookup[i][0], tf_lookup[i][1], tf_lookup[i][2])
            cassettes.append(Cassette("4", layer_2_tf))

    # Layer 3
    cassettes.append(Cassette("5", TranscriptionFactor("O", "", "")))

    tp_graph = TPGraph(input_ligands=rcds, init_root=False)
    tp_graph.root, n_roots = tp_graph_constructor_sepa(cassettes,verbose=False)
    
    return tp_graph, n_roots


def graph_to_string(this_node, this_string):
    if type(this_node) == ConstitutiveNode:
        _output_string = this_node.output.__str__().split()
        return "{}_{}^{}, ".format(_output_string[1][:-1], _output_string[0][:-1], _output_string[2])
    else:
        this_substring = ""
        for parent in this_node.parents:
            this_substring += graph_to_string(parent, this_string)
        if this_node.target_protein.__str__() == "O, , ":
            this_substring += "<O>"
            this_substring = "<" + this_substring + ">"
        else:
            _output_string = this_node.target_protein.__str__().split()
            this_substring += "<{}_{}^{}>".format(_output_string[1][:-1], _output_string[0][:-1], _output_string[2])
            this_substring = "<" + this_substring + ">, "
        this_string += this_substring
        return this_string


def csr_to_circuit_string(csr_rep):

    dense_rep = csr_rep.A
    tp_graph, n_roots = graph_from_matrix(dense_rep)

    if n_roots != 1:
        raise ValueError("n_roots must be 1")
    
    logic_str = ""
    for inp in three_inp_eval:
        logic_str += str(tp_graph.evaluate(inp))

    this_node = tp_graph.root
    this_string = ""
    this_string = graph_to_string(this_node, this_string)

    return logic_str, this_string
