import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from structures.nodes import *
from structures.transcription_factors import *
from structures.transcriptional_programs import *
import visualization.visualization_funcs as vis


class TPGraph:
    '''
    Wrapper for TPNode which represents the root node
    in the computational graph defining a transcriptional
    program
    '''
    def __init__(self, cassettes=None, input_ligands=None, init_root=True, verbose=False):
        '''
        Attributes
        ----------
        cassettes (list of Cassette) : the Cassettes from which
            the computational graph may be automatically constructed
        root (TPNode) : the automatically determined root node
            of the resultant computational graph
        input_ligands (list of str) : contains ligands which define
            the Boolean inputs to the computational graph
        '''
        if init_root:
            self.root, self.n_roots = tp_graph_constructor(cassettes, verbose=verbose)
        self.input_ligands = input_ligands

    def evaluate(self, logic_input):
        '''
        Evaluate computational graph using n-input
        boolean signal

        Parameters
        ----------
        logic_input (list of ints) : must be the same size as self.input_ligands

        Returns
        -------
        logic_output (int) : logical output of graph
        '''
        if len(logic_input) != len(self.input_ligands):
            print("Input length must match number of ligands!")
            return
        return self.root.logical_output(list(np.array(self.input_ligands)[np.array(logic_input,dtype=bool)]))

    def print_truth_table(self):
        '''
        Evaluate for all possible input signals
        and print the resultant truth table
        '''
        inp_lists = [[0,1] for _ in range(len(self.input_ligands))]
        for inp in itertools.product(*inp_lists):
            print(str(list(inp))+" "+str(self.evaluate(list(inp))))

    def categorize_circuit(self):
        '''
        Evaluate for all possible input signals
        and categorize the resultant truth table

        Returns
        -------
        category (str) : name of the gate represented
            by the computational graph
        '''
        if len(self.input_ligands) != 2:
            print("Only implemented for  2-input circuits!")
            return
        inp_lists = [[0,1] for _ in range(len(self.input_ligands))]
        outs = []
        for inp in itertools.product(*inp_lists):
            outs.append(self.evaluate(list(inp)))
        self.category = two_input_names[tuple(outs)]
        return self.category

    def compress_inversion(self, verbose=False):
        '''
        Replace all instances of inversion with
        a single anti-repressor basic program node.
        Re-wire the graph and reset the root

        Parameters
        ----------
        verbose (bool) : print the nodes as they are added
        '''
        retained_nodes = []
        removed_nodes = []
        replace_inversion(self.root, retained_nodes, removed_nodes, verbose=verbose)

        # identify the parents of each node
        # parents produce TFs which are used as inputs to a node's program
        for i in range(len(retained_nodes)):
            if isinstance(retained_nodes[i], ConstitutiveNode):
                continue
            retained_nodes[i].parents = []
            for j in range(len(retained_nodes)):
                if isinstance(retained_nodes[j], ConstitutiveNode):
                    out_j = retained_nodes[j].output
                else:
                    out_j = retained_nodes[j].program.output
                if out_j in retained_nodes[i].program.transcription_factors:
                    retained_nodes[i].parents.append(retained_nodes[j])

        # find the node which has no children
        n_children_dict = {}
        n_roots = 0
        for node in retained_nodes:
            n_children_dict[node] = 0
            for _node in retained_nodes:
                if (not isinstance(_node, ConstitutiveNode)) and node in _node.parents:
                    n_children_dict[node] += 1
            if n_children_dict[node] == 0:
                n_roots += 1
                if verbose:
                    print('root found')
                self.root = node

        # it is not good if there is anything but a single root
        if n_roots != 1:
            print("There are {} roots!".format(n_roots))
        self.n_roots = n_roots

    def render(self, hscale=2.5, vscale=1., xlim=[-5,5], title=None, savepath=None, show=True, label_font_size=12):
        '''
        Plot the computational graph using networkx

        Parameters
        ----------
        hscale (float) : how should each layer be spread
            around the layer below it, horizontally
        vscale (float) : how much space between each layer
        xlim (list of float) : limits of the plot
        title (str) : optional title of the graph
        savepath (str) : optional directory to save the plote
        show (bool) : indicate whether or not to display the plot
        label_font_size (int) : font size of labels on graph
        '''
        pos_dict = {}
        pos_dict[self.root.__str__()] = [0,0]
        
        vis.graph_node_pos(self.root, pos_dict, hscale, vscale)
        nodes, edges = vis.tp_graph_to_node_edges(self.root)
        nx_graph = vis.tp_graph_to_nx(nodes, edges)

        fig = plt.figure(figsize=(8,6))
        ax = plt.gca()
        nx.draw_networkx(nx_graph, pos=pos_dict, with_labels=True, font_size=label_font_size, ax=ax,
                         node_color='steelblue', edgecolors='steelblue', node_size=300)
        plt.xlim(xlim[0],xlim[1])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if title is not None:
            plt.title(title,fontsize=16)
        if savepath is not None:
            plt.savefig(savepath)
        if show:
            plt.show()
        plt.clf()
        plt.close()


def tp_graph_constructor(cassettes, verbose=False):
    '''
    Parameters
    ----------
    cassettes (list of Cassette) : the Cassettes from which
        the computational graph may be automatically constructed
    verbose (bool) : print the nodes as they are added

    Returns
    -------
    root (TPNode) : the automatically determined root node
        of the resultant computational graph
    '''


    # identify which pairs of cassettes can be connected together
    # this happens if the promotor of cassette can be paired with the output (dbd) of _cassette
    tp_nodes_init = []
    for cassette in cassettes:
        if cassette.promoter == "constitutive":
            tp_nodes_init.append(ConstitutiveNode(cassette.transcription_factor))
        else:
            used_tfs = []
            for _cassette in cassettes:
                if _cassette.transcription_factor.dbd == cassette.promoter:
                    if _cassette.transcription_factor not in used_tfs:
                        tp_nodes_init.append(TPNode(program=SEPAProgram([_cassette.transcription_factor],output=cassette.transcription_factor), parents=[]))
                        used_tfs.append(_cassette.transcription_factor)
                    

    # if multiple nodes output GFP, merge them into a MIMO node
    tp_nodes = []
    processed_idx = []
    for i in range(len(tp_nodes_init)):
        if i in processed_idx:
            continue
        elif (not isinstance(tp_nodes_init[i], ConstitutiveNode)) and (tp_nodes_init[i].program.output.dbd == "GFP" or tp_nodes_init[i].program.output.dbd == "O"):
            multi_inputs = [tp_nodes_init[i].program.transcription_factors[0]]
            multi_outs = [tp_nodes_init[i].program.output]
            for j in range(len(tp_nodes_init)):
                if (i!=j) and (not isinstance(tp_nodes_init[j], ConstitutiveNode)) and (tp_nodes_init[j].program.output.dbd == "GFP" or tp_nodes_init[j].program.output.dbd == "O"):
                    multi_inputs.append(tp_nodes_init[j].program.transcription_factors[0])
                    multi_outs.append(tp_nodes_init[i].program.output)
                    processed_idx.append(j)

            if len(multi_inputs) == 1:
                gfp_node = TPNode(program=SEPAProgram(multi_inputs, output=multi_outs[0]), parents=[])
            else:
                gfp_node = TPNode(program=MIMOProgram(multi_inputs,multi_outs), parents=[])
            tp_nodes.append(gfp_node)
            if verbose:
                print(tp_nodes[-1])
        else:
            tp_nodes.append(tp_nodes_init[i])
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


    # it is not good if there is anything but a single root
    if n_roots != 1:
        print("There are {} roots!".format(n_roots))

    return root_node, n_roots


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


def replace_inversion(root, retained_nodes, removed_nodes, verbose=False):
    '''
    Move recursively through the graph to detect
    inversion and replace with anti-repression

    Parameters
    ----------
    root (TPNode) : source node
    retained_nodes (list of TPNode) : list of nodes which
        are included in the graph up to this level
    removed_nodes (list of TPNode) : list of nodes which
        should be removed due to being compressed or "flipped"
    verbose (bool) : print the nodes as they are added
    '''

    # if root is already removed, move on to its parents
    if root in removed_nodes:
        if not isinstance(root, ConstitutiveNode):
            for i in range(len(root.parents)):
                replace_inversion(root.parents[i], retained_nodes, removed_nodes, verbose)
    
    elif not isinstance(root, ConstitutiveNode):
        
        # consider root which is a MIMO program
        if isinstance(root.program, MIMOProgram):

            # see if we can replace TF inputs using anti-repression
            new_transcription_factors = []
            for i in range(len(root.program.transcription_factors)):

                # only change something if the root TF is a repressor
                if root.program.transcription_factors[i].repression in ["+", "S"]:

                    # iterate through parents, find which one produces this TF
                    for parent in root.parents:

                        # if the TF is constitutively expressed, change nothing
                        if isinstance(parent, ConstitutiveNode):
                            if root.program.transcription_factors[i] == parent.output:
                                new_transcription_factors.append(root.program.transcription_factors[i])
                                break

                        # if the parent is a MIMO program, change nothing
                        elif isinstance(parent.program, MIMOProgram):
                            if root.program.transcription_factors[i] in parent.program.output:
                                new_transcription_factors.append(root.program.transcription_factors[i])
                                break

                        # if the parent is a SEPA program, change if the parent is SISO with repression
                        elif isinstance(parent.program, SEPAProgram):
                            if root.program.transcription_factors[i] == parent.program.output:
                                
                                # check if corrresponding parent is a basic program with repression
                                if len(parent.program.transcription_factors) == 1 and parent.program.transcription_factors[0].repression in ["+", "S"]:

                                    # we found inversion! remove the parent, its function will be replaced later
                                    removed_nodes.append(parent)
                                    
                                    # make an anti-repressor version of the parent TF
                                    anti_tf = TranscriptionFactor(parent.program.transcription_factors[0].dbd,
                                                                      parent.program.transcription_factors[0].rcd,
                                                                      "A")

                                    # if the parent TF is constitutively expressed, replace it with constitutive anti-repressor
                                    for j in range(len(parent.parents)):
                                        if isinstance(parent.parents[j], ConstitutiveNode):
                                            removed_nodes.append(parent.parents[j])
                                            retained_nodes.append(ConstitutiveNode(anti_tf))
                                            if verbose:
                                                print(retained_nodes[-1])

                                    new_transcription_factors.append(anti_tf)
                                    break
                                
                                # if corresponding parent is not a basic program with repression, change nothing
                                else:
                                    new_transcription_factors.append(root.program.transcription_factors[i])
                                    break

                # if the root TF is not a repressor, change nothing
                else:
                    new_transcription_factors.append(root.program.transcription_factors[i])


            # new node has updated TF list if some of the inputs use inversion
            new_node = TPNode(program=MIMOProgram(transcription_factors=new_transcription_factors, output=root.program.output),
                                  parents=[])
            removed_nodes.append(root)
            retained_nodes.append(new_node)
            if verbose:
                print(retained_nodes[-1])
            
            # move on to parent nodes
            for i in range(len(root.parents)):
                replace_inversion(root.parents[i], retained_nodes, removed_nodes, verbose)

        # consider root which is a basic program (SISO) using a repressor and has one parent
        elif len(root.program.transcription_factors) == 1 \
        and root.program.transcription_factors[0].repression in ["+", "S"] \
        and len(root.parents) == 1:
            
            # its parent must also be a basic program using a repressor
            if (not isinstance(root.parents[0], ConstitutiveNode)) \
            and len(root.parents[0].program.transcription_factors) == 1 \
            and root.parents[0].program.transcription_factors[0].repression in ["+", "S"]:

                # make an anti-repressor version of the parent TF
                anti_tf = TranscriptionFactor(root.parents[0].program.transcription_factors[0].dbd,
                                                  root.parents[0].program.transcription_factors[0].rcd,
                                                  "A")

                # if the parent TF is constitutively expressed, replace it with constitutive anti-repressor
                for j in range(len(root.parents[0].parents)):
                    if isinstance(root.parents[0].parents[j], ConstitutiveNode):
                        removed_nodes.append(root.parents[0].parents[j])
                        retained_nodes.append(ConstitutiveNode(anti_tf))
                        if verbose:
                            print(retained_nodes[-1])

                # make a single new node to replace the root and its parent, using the anti-repressor
                new_node = TPNode(program=SEPAProgram(transcription_factors=[anti_tf], output=root.program.output),
                                  parents=[])
                removed_nodes.append(root)
                removed_nodes.append(root.parents[0])
                retained_nodes.append(new_node)
                if verbose:
                    print(retained_nodes[-1])
            
            # if its parent is not a basic program using a repressor, keep root node
            else:
                retained_nodes.append(root)
                if verbose:
                    print(retained_nodes[-1])
            
            # move on to parent node
            replace_inversion(root.parents[0], retained_nodes, removed_nodes, verbose)

        else:
            retained_nodes.append(root)
            if verbose:
                print(retained_nodes[-1])
            for i in range(len(root.parents)):
                replace_inversion(root.parents[i], retained_nodes, removed_nodes, verbose)
    
    # if this node represents constitutive expression, keep it and stop recursion
    else:
        retained_nodes.append(root)
        if verbose:
            print(retained_nodes[-1])




two_input_names = {
    (0, 0, 0, 1) : "A AND B",
    (1, 1, 1, 0) : "A NAND B",
    
    (0, 1, 1, 1) : "A OR B",
    (1, 0, 0, 0,) : "A NOR B",
    
    (0, 1, 1, 0) : "A XOR B",
    (1, 0, 0, 1) : "A XNOR B",
    
    (1, 1, 0, 1) : "A IMPLY B",
    (0, 0, 1, 0) : "A NIMPLY B",
    
    (1, 0, 1, 1) : "A CONVIMPLY B",
    (0, 1, 0, 0) : "A CONVNIMPLY B",
    
    (0, 0, 0, 0) : "A CONTR B",
    (1, 1, 1, 1) : "A TAUT B",
    
    (1, 0, 1, 0) : "A PROJ B",
    (0, 1, 0, 1) : "A NPROJ B",
    
    (1, 1, 0, 0) : "A CONVPROJ B",
    (0, 0, 1, 1) : "A CONVNPROJ B",
}