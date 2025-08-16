from datetime import datetime

import graph_tool.all as gt
import numpy as np
import math
import os

from tqdm import tqdm

import dataset
from kmeans_wikics import KMeansWikiCS


def calculate_gravity_factor(g, degrees):
    """
    Calculates the Gravity Factor for each node as defined in Eq (4).
    Assumes W_mn = 1.
    Since W_mn=1, g(vm, vn) depends only on vm.
    """
    g_v = g.new_vertex_property("double")
    closeness = gt.closeness(g)

    # Calculate sum of closeness, handling potential 0 closeness for disconnected nodes
    c_sum = sum(closeness[v] for v in g.vertices() if closeness[v] > 0)
    if c_sum == 0:
        print("Warning: Sum of closeness centrality is zero. Cannot compute Gravity Factor.")
        return None

    for v in g.vertices():
        deg = degrees[v]
        g_v[v] = np.log(1 + (deg / c_sum))  # W_mn = 1
    return g_v


def calculate_mean_gravity(g, g_v):
    """
    Calculates Mean Gravity for each edge as defined in Eq (5).
    """
    m_uv = g.new_edge_property("double")
    for e in g.edges():
        u, v = e.source(), e.target()
        m_uv[e] = (g_v[u] + g_v[v]) / 2.0
    return m_uv


def calculate_sim_direct(g, degrees):
    """
    Calculates the direct similarity for each edge as defined in Eq (8).
    Assumes W_mn = 1.
    """
    sim_dir = g.new_edge_property("double")
    for e in g.edges():
        u, v = e.source(), e.target()
        deg_u = degrees[u]
        deg_v = degrees[v]
        denominator = deg_u + deg_v - 1.0
        if denominator == 0:
            sim_dir[e] = 1.0
        else:
            sim_dir[e] = 1.0 / denominator
    return sim_dir


def get_shortest_path(g, source, target):
    """
    Finds a shortest path between source and target using graph-tool.
    Returns a list of vertices and a list of edges.
    """
    vlist, elist = gt.shortest_path(g, source, target)
    return [v for v in vlist], [e for e in elist]


def calculate_SAS_D_matrix(g, alpha, g_v, m_uv, sim_dir, pol_lean):
    """
    Calculates the Structure-Attribute Similarity Distance (SAS-D) matrix.
    """
    N = g.num_vertices()
    sas_matrix = np.full((N, N), -1.0)
    sim_S_matrix = np.zeros((N, N))
    sim_A_matrix = np.zeros((N, N))

    node_map = {int(v): i for i, v in enumerate(g.vertices())}
    reverse_node_map = {i: int(v) for i, v in enumerate(g.vertices())}

    edge_map = {}
    for e in g.edges():
        u, v = int(e.source()), int(e.target())
        edge_map[(u, v)] = sim_dir[e]
        edge_map[(v, u)] = sim_dir[e]

    print("Calculating Sim_A...")
    for i in range(N):
        for j in range(i, N):
            v_m = g.vertex(reverse_node_map[i])
            v_n = g.vertex(reverse_node_map[j])
            sim_A = 1.0 if pol_lean[v_m] == pol_lean[v_n] else 0.0
            sim_A_matrix[i, j] = sim_A_matrix[j, i] = sim_A

    print("Calculating Sim_S and SAS...")
    dist_map = gt.shortest_distance(g)

    for i in tqdm(range(N)):
        # if i % 100 == 0:
        #     print(f"  Processing node {i}/{N}...")
        v_m = g.vertex(reverse_node_map[i])
        for j in range(i, N):
            v_n = g.vertex(reverse_node_map[j])

            if i == j:
                sim_S_matrix[i, j] = 1.0
                sas_matrix[i, j] = alpha * 1.0 + (1.0 - alpha) * 1.0
                continue

            distance = dist_map[v_m][v_n]

            if distance == 1:
                edge = g.edge(v_m, v_n)
                if edge is None:
                    print(f"Error: Edge not found between {i} and {j} despite distance 1.")
                    sim_S = 0.0
                else:
                    sim_v = sim_dir[edge]
                    m_v = m_uv[edge]
                    sim_S = sim_v + m_v
            elif distance < N + 1:
                path_nodes, path_edges = get_shortest_path(g, v_m, v_n)

                # Ensure path_nodes are actual vertex objects for g_v lookup
                p_v = np.prod([g_v[g.vertex(node_id)] for node_id in path_nodes])

                sim_v = 1.0
                for l in range(len(path_nodes) - 1):
                    u_id, v_id = path_nodes[l], path_nodes[l + 1]
                    sim_l = edge_map.get((u_id, v_id))
                    if sim_l is None:
                        print(f"Error: Sim_dir not found for edge ({u_id}, {v_id}) in path.")
                        sim_v = 0
                        break
                    sim_v *= sim_l

                sim_S = sim_v + p_v
            else:  # Disconnected
                sim_S = 0.0

            sim_S_matrix[i, j] = sim_S_matrix[j, i] = sim_S

            beta = 1.0 - alpha
            sas = alpha * sim_S + beta * sim_A_matrix[i, j]
            sas_matrix[i, j] = sas_matrix[j, i] = sas

    print("Normalizing SAS matrix...")
    min_sas = np.min(sas_matrix)
    max_sas = np.max(sas_matrix)

    if max_sas == min_sas:
        norm_sas_matrix = np.zeros((N, N))
    else:
        norm_sas_matrix = (sas_matrix - min_sas) / (max_sas - min_sas)

    print("Calculating SAS-D matrix...")
    sas_d_matrix = np.full((N, N), float(N + 1)) # Initialize with a large value for disconnected pairs

    for i in range(N):
        v_m = g.vertex(reverse_node_map[i])
        for j in range(i, N):
            v_n = g.vertex(reverse_node_map[j])
            if dist_map[v_m][v_n] < N + 1: # Only consider connected nodes
                sas_d_matrix[i, j] = sas_d_matrix[j, i] = 1.0 - norm_sas_matrix[i, j]

    return sas_d_matrix


def sas_cluster_kmedoids(sas_d_matrix, k, max_iter=100):
    """
    Performs K-medoids clustering using the precomputed SAS-D matrix.
    """
    N = sas_d_matrix.shape[0]

    # Initialize medoids randomly
    medoid_indices = np.random.choice(N, k, replace=False)
    clusters = -1 * np.ones(N, dtype=int)

    print(f"Starting K-Medoids with K={k}...")

    for iteration in range(max_iter):
        print(f"  Iteration {iteration + 1}...")

        # Assign each point to the closest medoid
        new_clusters = np.argmin(sas_d_matrix[:, medoid_indices], axis=1)

        # Check for convergence
        if np.array_equal(new_clusters, clusters):
            print("  Converged.")
            break
        clusters = new_clusters

        # Update medoids
        new_medoid_indices = np.zeros(k, dtype=int)
        for i in range(k):
            cluster_nodes = np.where(clusters == i)[0]
            if len(cluster_nodes) == 0:
                # If a cluster becomes empty, re-initialize its medoid randomly
                new_medoid_indices[i] = np.random.choice(N)
                print(f"  Warning: Cluster {i} became empty. Re-initializing medoid.")
                continue

            # Find the node in the cluster that minimizes the sum of distances to other cluster nodes
            costs = np.sum(sas_d_matrix[cluster_nodes][:, cluster_nodes], axis=1)
            new_medoid_indices[i] = cluster_nodes[np.argmin(costs)]

        medoid_indices = new_medoid_indices

    print("K-Medoids finished.")
    return clusters, medoid_indices


def calculate_density(g, clusters):
    """
    Calculates the Density as defined in Eq (1) (interpreted).
    It measures the fraction of edges that lie within clusters.
    """
    total_edges = g.num_edges()
    if total_edges == 0:
        return 0.0

    intra_cluster_edges = 0
    node_map = {int(v): i for i, v in enumerate(g.vertices())}

    for e in g.edges():
        u, v = e.source(), e.target()
        u_idx = node_map[int(u)]
        v_idx = node_map[int(v)]

        if clusters[u_idx] == clusters[v_idx]:
            intra_cluster_edges += 1

    return intra_cluster_edges / total_edges


def calculate_entropy(g, clusters, pol_lean):
    """
    Calculates the Entropy based on Eq (2) & (3) (interpreted).
    It measures the weighted average attribute entropy across clusters.
    """
    N = g.num_vertices()
    node_map = {int(v): i for i, v in enumerate(g.vertices())}
    reverse_node_map = {i: v for v, i in node_map.items()}

    k = len(np.unique(clusters))
    total_entropy = 0.0

    for cluster_id in range(k):
        cluster_nodes_indices = np.where(clusters == cluster_id)[0]
        num_nodes_in_cluster = len(cluster_nodes_indices)

        if num_nodes_in_cluster == 0:
            continue

        unique_labels = set(pol_lean[v] for v in g.vertices())
        counts = {label: 0 for label in unique_labels}

        for idx in cluster_nodes_indices:
            node = g.vertex(reverse_node_map[idx])
            counts[pol_lean[node]] += 1

        cluster_entropy = 0.0
        for value in counts:
            p = counts[value] / num_nodes_in_cluster
            if p > 0:
                cluster_entropy -= p * math.log2(p)

        weight = num_nodes_in_cluster / N
        total_entropy += weight * cluster_entropy

    return total_entropy


# --- Main Execution ---

dataset_name = "wikics"
g, pol_lean = dataset.load_graph(dataset_name)

# --- Parameter Setting ---
K = 30
ALPHA = 0.7
MAX_ITER = 100

# --- Pre-calculations ---
print("Performing pre-calculations...")
degrees = g.degree_property_map("total")

closeness_map = gt.closeness(g)
closeness_sum = sum(closeness_map[v] for v in g.vertices() if closeness_map[v] > 0)

if closeness_sum > 0:
    g_v = calculate_gravity_factor(g, degrees)

    if g_v:
        m_uv = calculate_mean_gravity(g, g_v)
        sim_dir = calculate_sim_direct(g, degrees)

        print("Calculating SAS-D Matrix (this may take a while)...")
        sas_d_matrix = calculate_SAS_D_matrix(g, ALPHA, g_v, m_uv, sim_dir, pol_lean)

        clusters, medoids = sas_cluster_kmedoids(sas_d_matrix, K, MAX_ITER)

        # --- Calculate Evaluation Metrics ---
        density = calculate_density(g, clusters)
        entropy = calculate_entropy(g, clusters, pol_lean)

        # --- Output Results ---
        print("\n--- Clustering Results ---")
        print(f"K = {K}, Alpha = {ALPHA}")
        print(f"Final Medoid Indices: {medoids}")

        unique, counts = np.unique(clusters, return_counts=True)
        print("Cluster Sizes:")
        for i in range(K):
            # Check if the cluster ID 'i' exists in 'unique'
            if i in unique:
                size = counts[np.where(unique == i)[0][0]]
            else:
                size = 0 # If a cluster ID is not present, it means it's empty
            print(f"  Cluster {i}: {size} nodes")

        print("\n--- Evaluation Metrics ---")
        print(f"Density: {density:.4f}")  #
        print(f"Entropy: {entropy:.4f}")  #

        # get current date and time string, e.g. "2024-04-27_15-30-12"
        dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # define folder path with date and time, e.g. out/2024-04-27_15-30-12/dataset_name
        folder_path = os.path.join("out", dt_string, dataset_name)

        # create directories if they don't exist
        os.makedirs(folder_path, exist_ok=True)

        # save clusters to a file under this folder
        filename = os.path.join(folder_path, f"clusters_alpha_{ALPHA}_K_{K}.npy")

        with open(filename, 'wb') as f:
            np.save(f, clusters)

        print(f"Clusters saved to {filename}")

    else:
        print("Could not calculate Gravity Factor. Aborting.")
else:
    print("Closeness sum is zero, cannot proceed. Is the graph empty or entirely disconnected?")
