import numpy as np
import networkx as nx
from structures.nodes import *

def tp_graph_to_node_edges(root):
    '''
    for visualization, return node names and
    edges between nodes by traversing the pre-defined
    computational graph
    
    Parameters
    ----------
    root (TPNode) : root node of graph (e.g., GFP expression)
    
    Returns
    -------
    nodes (list of str) : names of the nodes in the graph
    edges (list of list) : each list contains source to target node
    '''
    nodes = [root.__str__()]
    edges = []
    for parent in root.parents:
        if isinstance(parent, ConstitutiveNode):
            continue
        parent_nodes, parent_edges = tp_graph_to_node_edges(parent)
        nodes += parent_nodes
        edges += parent_edges
        edges.append([parent.__str__(), root.__str__()])
    return nodes, edges


def graph_node_pos(root, pos_dict, hscale, vscale):
    '''
    for visualization, update dictionary of positions
    to recursively add positions of all parents
    
    Parameters
    ----------
    root (TPNode) : root node of graph (e.g., GFP expression)
    pos_dict (dict) : positions of nodes including root
        and its children
    hscale (float) : how should the next layer up be spread
        around the root, horizontally
    vscale (float) : how much higher up should the next layer be
    '''
    if not isinstance(root, ConstitutiveNode):
        n_parents = len(root.parents)
        root_pos = pos_dict[root.__str__()]
        if n_parents != 1:
            horiz_positions = np.linspace(root_pos[0]-hscale, root_pos[0]+hscale, n_parents)
        else:
            horiz_positions = np.array([root_pos[0]])

        for i in range(n_parents):
            pos_dict[root.parents[i].__str__()] = [horiz_positions[i], root_pos[1]+vscale]
            graph_node_pos(root.parents[i], pos_dict, hscale/2, vscale)


def tp_graph_to_nx(nodes, edges):
    '''
    for visualization, convert nodes and edges
    to a networkx graph
    
    Parameters
    ----------
    nodes (list of str) : names of the nodes in the graph
    edges (list of list) : each list contains source to target node
    
    Returns
    -------
    nx_graph (networkx DiGraph) : graph object for plotting purposes
    '''
    nx_graph = nx.DiGraph()
    for node in nodes:
        nx_graph.add_node(node)
    for edge in edges:
        nx_graph.add_edge(edge[0],edge[1])
    return nx_graph