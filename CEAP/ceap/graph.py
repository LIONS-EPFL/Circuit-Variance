from typing import List, Dict, Union, Tuple, Literal, Optional, Set, Sequence
import os
import json
import heapq

import torch
from transformer_lens import HookedTransformerConfig
from transformer_lens.hook_points import HookedRootModule
import numpy as np
import pygraphviz as pgv
import warnings

from scipy.stats import hypergeom
from numpy.typing import NDArray

from .utils import EDGE_TYPE_COLORS, generate_random_color

class Node:
    """
    A node in our computational graph. The in_hook is the TL hook into its inputs, 
    while the out_hook gets its outputs.
    """
    name: str
    layer: int
    in_hook: str
    out_hook: str
    index: Tuple
    parents: Set['Node']
    parent_edges: Set['Edge']
    children: Set['Node']
    child_edges: Set['Edge']
    in_graph: bool
    qkv_inputs: Optional[List[str]]
    compute_layer: int # It is different from layer. Nodes that are in the same compute_layer are necessary and sufficient to produce the output.

    def __init__(self, name: str, layer:int, in_hook: List[str], out_hook: str, index: Tuple, qkv_inputs: Optional[List[str]]=None):
        self.name = name
        self.layer = layer
        self.in_hook = in_hook
        self.out_hook = out_hook 
        self.index = index
        self.in_graph = True 
        self.parents = set()
        self.children = set()
        self.parent_edges = set()
        self.child_edges = set()
        self.qkv_inputs = qkv_inputs
        self.compute_layer = None


    def __eq__(self, other):
        return self.name == other.name
    
    def __repr__(self):
        return f'Node({self.name}, in_graph: {self.in_graph})'
    
    def __hash__(self):
        return hash(self.name)

class LogitNode(Node):
    def __init__(self, n_layers:int):
        name = 'logits' 
        index = slice(None) 
        super().__init__(name, n_layers - 1, f"blocks.{n_layers - 1}.hook_resid_post", '', index)
        
class MLPNode(Node):
    def __init__(self, layer: int):
        name = f'm{layer}' 
        index = slice(None) 
        super().__init__(name, layer, f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out", index)

class AttentionNode(Node):
    head: int
    def __init__(self, layer:int, head:int):
        name = f'a{layer}.h{head}' 
        self.head = head
        index = (slice(None), slice(None), head) 
        # notice that all the AttentionNodes (corresponding to all heads) share the same in/out/qkv hooks.
        super().__init__(name, layer, f'blocks.{layer}.hook_attn_in', f"blocks.{layer}.attn.hook_result", index, [f'blocks.{layer}.hook_{letter}_input' for letter in 'qkv'])

class InputNode(Node):
    def __init__(self):
        name = 'input' 
        index = slice(None) 
        super().__init__(name, 0, '', "hook_embed", index) 

class Edge:
    name: str
    parent: Node 
    child: Node 
    hook: str
    index: Tuple
    score: Optional[NDArray[np.float_]]
    in_graph: NDArray[np.bool_]
    def __init__(self, parent: Node, child: Node, qkv:Union[None, Literal['q'], Literal['k'], Literal['v']]=None):
        self.name = f'{parent.name}->{child.name}' if qkv is None else f'{parent.name}->{child.name}<{qkv}>'
        self.parent = parent 
        self.child = child
        self.qkv = qkv
        # The datatype used below are key to enable in-place update to inform all the involved compute layers when scoring one edge.
        self.score = np.array(np.NaN,dtype=float) 
        self.in_graph = np.array(True)
        if isinstance(child, AttentionNode):
            if qkv is None:
                raise ValueError(f'Edge({self.name}): Edges to attention heads must have a non-none value for qkv.')
            self.hook = f'blocks.{child.layer}.hook_{qkv}_input'
            self.index = (slice(None), slice(None), child.head)
        else:
            self.index = child.index
            self.hook = child.in_hook
    def get_color(self):
        if self.qkv is not None:
            return EDGE_TYPE_COLORS[self.qkv]
        elif self.score < 0:
            return "#FF0000"
        else:
            return "#000000"

    def __eq__(self, other):
        return self.name == other.name
    
    def __repr__(self):
        return f'Edge({self.name}, score: {self.score}, in_graph: {self.in_graph})'
    
    def __hash__(self):
        return hash(self.name)

class Graph:
    nodes: Dict[str, Node]
    edges: Dict[str, Edge]
    n_forward: int 
    n_backward: int
    cfg: HookedTransformerConfig

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.n_forward = 0
        self.n_backward = 0
        # the following three are for tracking scores in each layer.
        self.layer_edges = []
        self.layer_edge_scores = []
        self.layer_edge_in_graph = []

    def add_edge(self, parent:Node, child:Node, qkv:Union[None, Literal['q'], Literal['k'], Literal['v']]=None):
        edge = Edge(parent, child, qkv)
        self.edges[edge.name] = edge
        parent.children.add(child)
        parent.child_edges.add(edge)
        child.parents.add(parent)
        child.parent_edges.add(edge)
        start_compute_layer, end_compute_layer = parent.compute_layer, child.compute_layer
        for l in range(start_compute_layer, end_compute_layer): # l can be understood as the layer index whose output we are looking at
            self.layer_edges[l].append(edge)
            self.layer_edge_scores[l].append(edge.score)
            self.layer_edge_in_graph[l].append(edge.in_graph)

    def forward_index(self, node:Node, attn_slice=True):
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            return self.n_forward
        elif isinstance(node, MLPNode):
            return 1 + node.layer * (self.cfg['n_heads'] + 1) + self.cfg['n_heads']
        elif isinstance(node, AttentionNode):
            i =  1 + node.layer * (self.cfg['n_heads'] + 1)
            return slice(i, i + self.cfg['n_heads']) if attn_slice else i + node.head # if attn_slice, then all the heads in that layer would return the same slice.
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")
        

    def backward_index(self, node:Node, qkv=None, attn_slice=True):
        if isinstance(node, InputNode):
            raise ValueError(f"No backward for input node")
        elif isinstance(node, LogitNode):
            return -1
        elif isinstance(node, MLPNode):
            return (node.layer) * (3 * self.cfg['n_heads'] + 1) + 3 * self.cfg['n_heads']
        elif isinstance(node, AttentionNode):
            assert qkv in 'qkv', f'Must give qkv for AttentionNode, but got {qkv}'
            i = node.layer * (3 * self.cfg['n_heads'] + 1) + ('qkv'.index(qkv) * self.cfg['n_heads'])
            return slice(i, i + self.cfg['n_heads']) if attn_slice else i + node.head
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    def scores(self, nonzero=False, in_graph=False, sort=True):
        if sort: print("scores are sorted!")
        s = torch.tensor([edge.score.item() for edge in self.edges.values() if edge.score != 0 and (edge.in_graph or not in_graph)]) if nonzero else torch.tensor([edge.score.item() for edge in self.edges.values()])
        return torch.sort(s).values if sort else s
    
    def selected_edge_ids(self):
        edge_in_graph_list = [edge.in_graph.item() for edge in self.edges.values()]
        return torch.tensor(
            [i for i, in_graph in enumerate(edge_in_graph_list) if in_graph],
            dtype=torch.long,
        )

    def count_included_edges(self):
        return sum(edge.in_graph for edge in self.edges.values())
    
    def count_included_nodes(self):
        return sum(node.in_graph for node in self.nodes.values())

    def apply_threshold(self, threshold: float, absolute: bool):
        threshold = float(threshold)
        for node in self.nodes.values():
            node.in_graph = True 
            
        for edge in self.edges.values():
            edge.in_graph[...] = abs(edge.score) >= threshold if absolute else edge.score >= threshold
    
    def apply_topn(self, n:int, absolute: bool = True):
        a = abs if absolute else lambda x: x
        for node in self.nodes.values(): 
            node.in_graph = False

        sorted_edges = sorted(list(self.edges.values()), key = lambda edge: a(edge.score), reverse=True)
        for edge in sorted_edges[:n]:
            edge.in_graph[...] = True 
            edge.parent.in_graph = True 
            edge.child.in_graph = True 

        for edge in sorted_edges[n:]:
            edge.in_graph[...] = False

    def apply_greedy(self, n_edges_target, n_edges_already=0, reset=True, absolute: bool=True):
        """Select a connected circuit greedily by edge score.

        Starting from the logits node when `reset` is true, repeatedly add the
        highest-scoring available edge whose child is already in the circuit. When
        a new parent node is added, its incoming edges become candidates for later
        steps. If `absolute` is true, rank edges by absolute score.
        """
        # Warning, if all edges are already in graph, then there might be degenerate behavior
        assert n_edges_target > n_edges_already

        if reset:
            if n_edges_already!=0:
                n_edges_already = 0
                warnings.warn("reset is True, setting n_edges_already to zero.")
            for node in self.nodes.values():
                node.in_graph = False 
            for edge in self.edges.values():
                edge.in_graph[...] = False
            self.nodes['logits'].in_graph = True

        if not reset:
            assert self.count_included_edges() == n_edges_already, "Error of n_edges_already."

        n_edges_to_add = n_edges_target - n_edges_already

        def abs_id(s: float):
            return abs(s) if absolute else s

        candidate_edges = sorted([edge for edge in self.edges.values() if edge.child.in_graph and not edge.in_graph[...]], key = lambda edge: abs_id(edge.score), reverse=True)
        # Next line seems to have no use. Candidate edges are already sorted. I just kept it there because the original repo did.
        edges = heapq.merge(candidate_edges, key = lambda edge: abs_id(edge.score), reverse=True)
        while n_edges_to_add > 0:
            n_edges_to_add -= 1
            top_edge = next(edges) 
            top_edge.in_graph[...] = True 
            parent = top_edge.parent
            if not parent.in_graph:
                parent.in_graph = True
                parent_parent_edges = sorted([parent_edge for parent_edge in parent.parent_edges], key = lambda edge: abs_id(edge.score), reverse=True)
                edges = heapq.merge(edges, parent_parent_edges, key = lambda edge: abs_id(edge.score), reverse=True)

    def prune_dead_nodes(self, prune_childless=True, prune_parentless=True):
        """Remove nodes and edges that cannot participate in the selected circuit.

        Childless pruning removes nodes whose outputs do not feed any selected
        edge, optionally removing their incoming edges as well. Parentless pruning
        removes non-input nodes that are still selected but have no selected input
        edge, along with their outgoing edges.
        """
        self.nodes['logits'].in_graph = any(parent_edge.in_graph for parent_edge in self.nodes['logits'].parent_edges)

        # initialize node.in_graph + prune childless
        for node in reversed(self.nodes.values()):
            if isinstance(node, LogitNode):
                continue 
            if any(child_edge.in_graph for child_edge in node.child_edges) : # if its output matters
                node.in_graph = True
            else:
                node.in_graph = False
                if prune_childless:
                    for parent_edge in node.parent_edges:
                        parent_edge.in_graph[...] = False

        # prune parentless
        if prune_parentless:
            for node in self.nodes.values():
                if not isinstance(node, InputNode) and node.in_graph and not any(parent_edge.in_graph for parent_edge in node.parent_edges):
                    node.in_graph = False 
                    for child_edge in node.child_edges:
                        child_edge.in_graph[...] = False # This mechanism also makes sure that there is no childless node. The removed edges are always the children of a removed nodes.


    @classmethod
    def from_model(cls, model_or_config: Union[HookedRootModule, HookedTransformerConfig, Dict]):
        """Construct a CEAP graph from a TransformerLens model, config, or config dict.

        The graph contains an input node, one attention node per head, one MLP node
        per layer, and a final logits node. Edges follow the residual-stream
        dependencies implied by the model configuration, including whether
        attention and MLP blocks run in parallel.
        """
        graph = Graph()
        if isinstance(model_or_config, HookedTransformerConfig):
            cfg = model_or_config
            graph.cfg = {'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp': cfg.parallel_attn_mlp}
        elif isinstance(model_or_config, dict):
            graph.cfg = model_or_config
        elif hasattr(model_or_config, "cfg"):
            cfg = model_or_config.cfg
            graph.cfg = {
                'n_layers': cfg.n_layers,
                'n_heads': cfg.n_heads,
                # Sparse adapter models follow the non-parallel block structure and
                # do not expose TL's parallel_attn_mlp flag.
                'parallel_attn_mlp': getattr(cfg, 'parallel_attn_mlp', False),
            }
        else:
            raise TypeError(f"Unsupported model/config type for Graph.from_model: {type(model_or_config)!r}")
        
        input_node = InputNode()
        graph.nodes[input_node.name] = input_node
        residual_stream = [input_node]
        # No need to update compute layer information. The input of the input_node we don't care about.
        input_node.compute_layer = 0

        for layer in range(graph.cfg['n_layers']):
            attn_nodes = [AttentionNode(layer, head) for head in range(graph.cfg['n_heads'])]
            mlp_node = MLPNode(layer)
            
            for attn_node in attn_nodes: 
                graph.nodes[attn_node.name] = attn_node 
            graph.nodes[mlp_node.name] = mlp_node     
                                    
            if graph.cfg['parallel_attn_mlp']: # All previous nodes connect to afterward nodes
                graph.add_an_empty_layer()
                for attn_node in attn_nodes:          
                    attn_node.compute_layer = len(graph.layer_edge_scores) # Compute layer information does not contain the input node
                    for node in residual_stream:
                        for letter in 'qkv':           
                            graph.add_edge(node, attn_node, qkv=letter)
                
                mlp_node.compute_layer = len(graph.layer_edge_scores)
                for node in residual_stream:
                    graph.add_edge(node, mlp_node)
                
                residual_stream += attn_nodes
                residual_stream.append(mlp_node)

            else: 
                graph.add_an_empty_layer()
                for attn_node in attn_nodes:     
                    attn_node.compute_layer = len(graph.layer_edge_scores) # Compute layer information does not contain the input node
                    for node in residual_stream:
                        for letter in 'qkv':           
                            graph.add_edge(node, attn_node, qkv=letter) 
                residual_stream += attn_nodes

                graph.add_an_empty_layer()
                mlp_node.compute_layer = len(graph.layer_edge_scores)
                for node in residual_stream:
                    graph.add_edge(node, mlp_node)
                residual_stream.append(mlp_node)

        logit_node = LogitNode(graph.cfg['n_layers'])
        graph.add_an_empty_layer()
        logit_node.compute_layer = len(graph.layer_edge_scores)
        for node in residual_stream:
            graph.add_edge(node, logit_node)
            
        graph.nodes[logit_node.name] = logit_node

        graph.n_forward = 1 + graph.cfg['n_layers'] * (graph.cfg['n_heads'] + 1)
        graph.n_backward = graph.cfg['n_layers'] * (3 * graph.cfg['n_heads'] + 1) + 1

        return graph
    
    @classmethod
    def intersection(cls,intersection_graph, graph_list):
        """
        Updates ``intersection_graph`` so that its ``in_graph`` flags represent the
        intersection of all graphs provided in ``graph_list``. Scores and other
        metadata on ``intersection_graph`` remain unchanged.

        Args:
            intersection_graph: Graph instance to update in-place.
            graph_list: Iterable of Graphs whose in-graph membership should be
                intersected. Must not include ``intersection_graph``.
        """
        graph_list = list(graph_list)
        if not graph_list:
            raise ValueError("graph_list must contain at least one graph to intersect.")

        # Ensure every graph has the necessary nodes/edges before updating.
        node_keys = set(intersection_graph.nodes.keys())
        edge_keys = set(intersection_graph.edges.keys())
        for idx, graph in enumerate(graph_list): 
            if set(graph.nodes.keys()) != node_keys:
                missing = node_keys.symmetric_difference(graph.nodes.keys())
                raise ValueError(f"Graph at index {idx} has mismatched node set; diff: {missing}.")
            if set(graph.edges.keys()) != edge_keys:
                missing = edge_keys.symmetric_difference(graph.edges.keys())
                raise ValueError(f"Graph at index {idx} has mismatched edge set; diff: {missing}.")

        for name, node in intersection_graph.nodes.items():
            node.in_graph = all(graph.nodes[name].in_graph for graph in graph_list)

        for name, edge in intersection_graph.edges.items():
            edge.in_graph[...] = all(graph.edges[name].in_graph for graph in graph_list)

        return intersection_graph


    def to_json(self, filename):
        # non serializable info
        d = {'cfg':self.cfg, 'nodes': {str(name): bool(node.in_graph) for name, node in self.nodes.items()}, 'edges':{str(name): {'score': None if edge.score is None or edge.score == np.NaN else float(edge.score), 'in_graph': bool(edge.in_graph)} for name, edge in self.edges.items()}}
        path = os.fspath(filename)
        directory = os.path.dirname(path)
        if directory: 
            os.makedirs(directory, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(d, f)

    @classmethod
    def from_json(cls, filename, return_dict = False):
        with open(filename, 'r') as f:
            d = json.load(f)
        if return_dict:
            return d # instantiating the graph below is time-consuming. Sometimes it suffices to just work with the dictionary.
        g = Graph.from_model(d['cfg'])
        for name, in_graph in d['nodes'].items():
            g.nodes[name].in_graph = in_graph
        
        for name, info in d['edges'].items():
            g.edges[name].score[...] = info['score']
            g.edges[name].in_graph[...] = info['in_graph']

        return g
    
    def __eq__(self, other):
        keys_equal = (set(self.nodes.keys()) == set(other.nodes.keys())) and (set(self.edges.keys()) == set(other.edges.keys()))
        if not keys_equal:
            return False
        
        for name, node in self.nodes.items():
            if node.in_graph != other.nodes[name].in_graph:
                return False 
            
        for name, edge in self.edges.items():
            if (edge.in_graph != other.edges[name].in_graph) or not np.allclose(edge.score, other.edges[name].score):
                return False
        return True

    def to_graphviz(
        self,
        colorscheme: str = "Pastel2",
        minimum_penwidth: float = 0.6,
        maximum_penwidth: float = 5.0,
        layout: str="dot",
        seed: Optional[int] = None
    ) -> pgv.AGraph:
        """
        Colorscheme: a cmap colorscheme
        """
        g = pgv.AGraph(directed=True, bgcolor="white", overlap="false", splines="true", layout=layout)

        if seed is not None:
            np.random.seed(seed)

        colors = {node.name: generate_random_color(colorscheme) for node in self.nodes.values()}

        for node in self.nodes.values():
            if node.in_graph:
                g.add_node(node.name, 
                        fillcolor=colors[node.name], 
                        color="black", 
                        style="filled, rounded",
                        shape="box", 
                        fontname="Helvetica",
                        )

        scores = self.scores().abs()
        # Do some sanity check. graphviz should only be used if attribute function is already applied
        if torch.isnan(scores).any():
            raise ValueError("NaN detected in edge scores — please run attribute() first.")
        if any(s is None for s in scores.tolist()):
            raise ValueError("None detected in edge scores — please run attribute() first.")
        max_score = scores.max().item()
        min_score = scores.min().item()
        for edge in self.edges.values():
            if edge.in_graph:
                score = 0 if (edge.score == np.array(None) or edge.score == np.NaN).item() else edge.score.item()
                normalized_score = (abs(score) - min_score) / (max_score - min_score) if max_score != min_score else abs(score)
                penwidth = max(minimum_penwidth, normalized_score * maximum_penwidth)
                g.add_edge(edge.parent.name,
                        edge.child.name,
                        penwidth=str(penwidth),
                        color=edge.get_color(),
                        )
        return g

    def add_an_empty_layer(self):
        """
        Adds an empty layer to the graph.
        """
        self.layer_edge_scores.append([])
        self.layer_edges.append([])
        self.layer_edge_in_graph.append([])


    def snapshot_selection_state(self):
        """Sometimes one wants to revert the pruning, which this function helps"""
        return {
            "nodes": {name: node.in_graph for name, node in self.nodes.items()},
            "edges": {name: edge.in_graph.copy() for name, edge in self.edges.items()},
        }

    def restore_selection_state(self, snapshot):
        """Sometimes one wants to revert the pruning, which this function helps"""
        for name, flag in snapshot["nodes"].items():
            self.nodes[name].in_graph = flag
        for name, mask in snapshot["edges"].items():
            self.edges[name].in_graph[...] = mask

    def mirror_graph(self, other: "Graph", *, copy_scores: bool = False):
        """
        Copy the node/edge in_graph flags from another graph with the same structure.

        Parameters
        ----------
        other:
            Reference graph whose in_graph flags will be copied.
        copy_scores:
            When True, also copy the edge scores. Useful when you want the new graph
            to start from the same attribution values as the reference graph.
        """
        self_nodes = set(self.nodes)
        other_nodes = set(other.nodes)
        if self_nodes != other_nodes:
            raise KeyError(
                "Graphs have mismatched node sets; "
                f"only in target: {sorted(self_nodes - other_nodes)}, "
                f"only in source: {sorted(other_nodes - self_nodes)}"
            )

        self_edges = set(self.edges)
        other_edges = set(other.edges)
        if self_edges != other_edges:
            raise KeyError(
                "Graphs have mismatched edge sets; "
                f"only in target: {sorted(self_edges - other_edges)}, "
                f"only in source: {sorted(other_edges - self_edges)}"
            )

        for name, node in other.nodes.items():
            self.nodes[name].in_graph = node.in_graph

        for name, edge in other.edges.items():
            self.edges[name].in_graph[...] = edge.in_graph
            if copy_scores:
                self.edges[name].score[...] = edge.score
