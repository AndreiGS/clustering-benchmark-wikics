import json
import graph_tool.all as gt
import pandas as pd


def load_polblogs():
    print("Loading Polblogs graph...")
    g = gt.collection.ns["polblogs"]
    g.set_directed(False)
    pol_lean = g.vp.value
    print(f"Graph loaded. Nodes: {g.num_vertices()}, Edges: {g.num_edges()}")
    return g, pol_lean

def load_news(nodes_filename, edges_filename):
    print("Loading News graph...")

    # Load node info
    news_info = pd.read_csv(nodes_filename)
    print(f"Loaded node info for {len(news_info)} nodes.")

    # Create a new graph
    g = gt.Graph(directed=False)

    # Dictionary to keep mapping from node ID to vertex
    node_id_to_vertex = {}

    # Add vertices and attach properties dynamically
    for col in news_info.columns:
        if col == "is_fake":
            g.vp[col] = g.new_vertex_property("int")
        else:
            g.vp[col] = g.new_vertex_property("string")

    # Add vertices
    for node_id, row in news_info.iterrows():
        v = g.add_vertex()
        node_id_to_vertex[node_id] = v
        for col in news_info.columns:
            g.vp[col][v] = str(row[col])

    # Load edges
    edges = pd.read_csv(edges_filename)
    for row in edges.itertuples(index=False):
        src = getattr(row, "_0")
        dst = getattr(row, "to")
        if src in node_id_to_vertex and dst in node_id_to_vertex:
            g.add_edge(node_id_to_vertex[src], node_id_to_vertex[dst])

    print(f"Graph loaded. Nodes: {g.num_vertices()}, Edges: {g.num_edges()}")
    return g, g.vp["is_fake"]

# def load_wikics_nolimit(filename, split='train', self_loop=False):
#     """
#     Load the WikiCS dataset.
#
#     Parameters:
#     -----------
#     filename : str
#         Path to the WikiCS JSON file
#     self_loop : bool, optional
#         Whether to add self loops to the graph
#
#     Returns:
#     --------
#     g : gt.Graph
#         The graph
#     labels : gt.VertexPropertyMap
#         The vertex property map with labels
#     """
#     print(f"Loading WikiCS graph from {filename}...")
#
#     # Load data from JSON file
#     with open(filename, 'r') as f:
#         data = json.load(f)
#
#     # Create a new graph
#     g = gt.Graph(directed=False)
#
#     # Add vertices
#     num_nodes = len(data['features'])
#     for _ in range(num_nodes):
#         g.add_vertex()
#
#     # Add edges
#     for i, neighbors in enumerate(data['links']):
#         for nb in neighbors:
#             g.add_edge(i, nb)
#
#     # Handle self loops as requested
#     if not self_loop:
#         gt.remove_self_loops(g)
#     else:
#         # First remove any existing self-loops to avoid duplicates
#         gt.remove_self_loops(g)
#         # Then add self-loops to all vertices
#         for v in g.vertices():
#             g.add_edge(v, v)
#
#     # Add labels as vertex property
#     labels = g.new_vertex_property("int")
#     for i, label in enumerate(data['labels']):
#         labels[i] = int(label)
#
#     # Store labels as a vertex property
#     g.vp["label"] = labels
#
#     print(f"Graph loaded. Nodes: {g.num_vertices()}, Edges: {g.num_edges()}")
#     print(f"Number of classes: {len(set(data['labels']))}")
#
#     return g, g.vp["label"]

# def load_wikics_nolimit(filename, split='train', self_loop=False):
#     """
#     Load the WikiCS dataset with split-specific node filtering.
#
#     Parameters:
#     -----------
#     filename : str
#         Path to the WikiCS JSON file
#     split : str, optional
#         Which split to load ('train', 'val', 'test')
#     self_loop : bool, optional
#         Whether to add self loops to the graph
#
#     Returns:
#     --------
#     g : gt.Graph
#         The graph (subgraph containing only nodes from the specified split)
#     labels : gt.VertexPropertyMap
#         The vertex property map with labels
#     """
#     print(f"Loading WikiCS graph from {filename} (split: {split})...")
#
#     # Load data from JSON file
#     with open(filename, 'r') as f:
#         data = json.load(f)
#
#     # Get the mask for the specified split
#     mask_key = f"{split}_masks"
#     if split == 'test':
#         mask_key = 'test_mask'
#     if mask_key not in data:
#         raise ValueError(
#             f"Split '{split}' not found. Available splits: {[k.replace('_masks', '') for k in data.keys() if k.endswith('_masks')]}, test")
#
#     # Get the mask for the specified split
#     split_mask = data[mask_key]
#     active_nodes = sum(split_mask)
#     num_nodes = len(data['features'])
#
#     print(f"Loading nodes based on {split} mask from {num_nodes} total nodes")
#     print(f"Split contains {active_nodes} active nodes")
#
#     # Create a new graph
#     g = gt.Graph(directed=False)
#
#     # Add all vertices (maintaining original indices)
#     g.add_vertex(num_nodes)
#
#     # Add edges only between nodes that are in the split
#     for i in range(num_nodes):
#         if not split_mask[i]:  # Only process nodes in the split
#             continue
#
#         neighbors = data['links'][i]
#         for nb in neighbors:
#             g.add_edge(i, nb, add_missing=False)
#
#     # Handle self loops as requested
#     if not self_loop:
#         gt.remove_self_loops(g)
#     else:
#         # First remove any existing self-loops to avoid duplicates
#         gt.remove_self_loops(g)
#         # Then add self-loops to all vertices
#         for v in g.vertices():
#             g.add_edge(v, v)
#
#     # Add labels as vertex property (for all nodes, but only split nodes will have edges)
#     labels = g.new_vertex_property("int")
#     for i in range(num_nodes):
#         labels[i] = int(data['labels'][i])
#
#     # Store labels as a vertex property
#     g.vp["label"] = labels
#
#     print(f"Graph loaded. Total nodes: {g.num_vertices()}, Edges: {g.num_edges()}")
#     print(f"Number of classes: {len(set(data['labels']))}")
#
#     return g, g.vp["label"]

def load_wikics_nolimit(filename, split='train', self_loop=False):
    """
    Load the WikiCS dataset with split-specific node filtering.

    Parameters:
    -----------
    filename : str
        Path to the WikiCS JSON file
    split : str, optional
        Which split to load ('train', 'val', 'test')
    self_loop : bool, optional
        Whether to add self loops to the graph

    Returns:
    --------
    g : gt.Graph
        The graph (subgraph containing only nodes from the specified split)
    labels : gt.VertexPropertyMap
        The vertex property map with labels
    """
    print(f"Loading WikiCS graph from {filename} (split: {split})...")

    # Load data from JSON file
    with open(filename, 'r') as f:
        data = json.load(f)

    # Get the mask for the specified split
    mask_key = f"{split}_masks"
    if split == 'test':
        mask_key = 'test_mask'
    if mask_key not in data:
        raise ValueError(
            f"Split '{split}' not found. Available splits: {[k.replace('_masks', '') for k in data.keys() if k.endswith('_masks')]}, test")

    # Get indices of nodes to include based on the mask
    split_mask = data[mask_key]
    selected_nodes = [i for i, include in enumerate(split_mask) if include]

    print(f"Selected {len(selected_nodes)} nodes from {len(split_mask)} total nodes for split '{split}'")

    # Create mapping from original node indices to new indices
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_nodes)}

    # Create a new graph
    g = gt.Graph(directed=False)

    # Add vertices (only for selected nodes)
    num_selected_nodes = len(selected_nodes)
    for _ in range(num_selected_nodes):
        g.add_vertex()

    # Add edges (only between selected nodes)
    for new_i, old_i in enumerate(selected_nodes):
        neighbors = data['links'][old_i]
        for old_nb in neighbors:
            # Only add edge if both nodes are in the selected set
            if old_nb in old_to_new:
                new_nb = old_to_new[old_nb]
                # Avoid duplicate edges (since graph is undirected)
                if new_i < new_nb:
                    g.add_edge(new_i, new_nb)

    # Handle self loops as requested
    if not self_loop:
        gt.remove_self_loops(g)
    else:
        # First remove any existing self-loops to avoid duplicates
        gt.remove_self_loops(g)
        # Then add self-loops to all vertices
        for v in g.vertices():
            g.add_edge(v, v)

    # Add labels as vertex property (only for selected nodes)
    labels = g.new_vertex_property("int")
    for new_idx, old_idx in enumerate(selected_nodes):
        labels[new_idx] = int(data['labels'][old_idx])

    # Store labels as a vertex property
    g.vp["label"] = labels

    print(f"Graph loaded. Nodes: {g.num_vertices()}, Edges: {g.num_edges()}")
    print(f"Number of classes: {len(set(labels[v] for v in g.vertices()))}")

    return g, g.vp["label"]

def load_wikics(filename, split='train', limit=None, self_loop=False):
    """
    Load the WikiCS dataset with reduced size (limit nodes).

    Parameters:
    -----------
    filename : str
        Path to the WikiCS JSON file
    self_loop : bool, optional
        Whether to add self loops to the graph

    Returns:
    --------
    g : gt.Graph
        The graph (reduced to 1000 nodes)
    labels : gt.VertexPropertyMap
        The vertex property map with labels
    """
    if limit is None:
        return load_wikics_nolimit(filename, split, self_loop)

    print(f"Loading WikiCS graph from {filename}...")

    # Load data from JSON file
    with open(filename, 'r') as f:
        data = json.load(f)

    # Limit to first limit nodes
    max_nodes = min(limit, len(data['features']))
    print(f"Reducing dataset to {max_nodes} nodes...")

    # Create a new graph
    g = gt.Graph(directed=False)

    # Add vertices (only first limit)
    for _ in range(max_nodes):
        g.add_vertex()

    # Add edges - only include edges where both endpoints are in range [0, max_nodes-1]
    edge_count = 0
    for i in range(max_nodes):  # Only iterate through first limit nodes
        if i < len(data['links']):
            for nb in data['links'][i]:
                # Only add edge if neighbor is also within our node range
                if nb < max_nodes:
                    g.add_edge(i, nb)
                    edge_count += 1

    print(f"Added {edge_count} edges within the reduced node set")

    # Handle self loops as requested
    if not self_loop:
        gt.remove_self_loops(g)
    else:
        # First remove any existing self-loops to avoid duplicates
        gt.remove_self_loops(g)
        # Then add self-loops to all vertices
        for v in g.vertices():
            g.add_edge(v, v)

    # Add labels as vertex property (only first 1000)
    labels = g.new_vertex_property("int")
    for i in range(max_nodes):
        if i < len(data['labels']):
            labels[i] = int(data['labels'][i])
        else:
            labels[i] = 0  # Default label if not enough labels in data

    # Store labels as a vertex property
    g.vp["label"] = labels

    # Get unique labels in the reduced dataset
    unique_labels = set()
    for i in range(max_nodes):
        if i < len(data['labels']):
            unique_labels.add(data['labels'][i])

    print(f"Reduced graph loaded. Nodes: {g.num_vertices()}, Edges: {g.num_edges()}")
    print(f"Number of classes in reduced dataset: {len(unique_labels)}")

    return g, g.vp["label"]


def load_graph(name):
    """
    Load a graph by name.

    Parameters:
    -----------
    name : str
        Name of the graph to load ("polblogs", "news", or "wikics")

    Returns:
    --------
    g : gt.Graph
        The graph
    labels : gt.VertexPropertyMap
        The vertex property map with labels
    """
    if name == "polblogs":
        return load_polblogs()
    elif name == "news":
        return load_news("./input/news/news_edges.csv", "./input/news/news_edges.csv")
    elif name.startswith("wikics"):
        return load_wikics("./input/wikics/data.json", split='test')
    else:
        raise ValueError(f"Unknown dataset: {name}")