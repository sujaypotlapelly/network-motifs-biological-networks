import networkx as nx
import itertools
import os
import numpy as np
from scipy import stats
import pandas as pd
import argparse
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import random
from tqdm import tqdm
from networkx.algorithms import isomorphism
from statsmodels.stats.multitest import multipletests

def setup_logging():
    """
    Configure logging to display time, level, and message.
    """
    logging.basicConfig(
        level=logging.INFO,  # Change to DEBUG for detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def load_network(file_path, network_type='directed', has_regulation=False):
    """
    Load a network from a tab-delimited file.

    Parameters:
    - file_path (str): Path to the data file.
    - network_type (str): 'directed' or 'undirected'.
    - has_regulation (bool): Whether the file includes regulation type.

    Returns:
    - G (networkx.Graph): The loaded graph.
    """
    if network_type == 'directed':
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    logging.info(f"Loading network from {file_path}...")
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')  # Ensure tab-delimited
            if has_regulation:
                if len(parts) < 3:
                    logging.warning(f"Line {line_num} is malformed: {line.strip()}")
                    continue  # Skip malformed lines
                regulator, target, regulation = parts[:3]
                G.add_edge(regulator, target, regulation=regulation)
            else:
                if len(parts) < 2:
                    logging.warning(f"Line {line_num} is malformed: {line.strip()}")
                    continue  # Skip malformed lines
                regulator, target = parts[:2]
                G.add_edge(regulator, target)
    logging.info(f"Loaded network '{file_path}' with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Debugging: Log sample nodes and edges
    if G.number_of_nodes() > 0:
        sample_nodes = list(G.nodes)[:5]
        sample_edges = list(G.edges)[:5]
        logging.info(f"Sample nodes: {sample_nodes}")
        logging.info(f"Sample edges: {sample_edges}")
    else:
        logging.error("The graph has no nodes. Please check the input file format.")

    return G

def get_canonical_form(subgraph):
    """
    Get a canonical string representation of a subgraph, including regulation types.

    Parameters:
    - subgraph (networkx.Graph): The subgraph.

    Returns:
    - str: Canonical string representation including regulation.
    """
    nodes = sorted(subgraph.nodes())
    mapping = {node: i for i, node in enumerate(nodes)}
    relabeled = nx.relabel_nodes(subgraph, mapping)
    
    if subgraph.is_directed():
        edge_list = sorted(relabeled.edges(data=True))
        edge_str = ';'.join([f"{u}->{v}:{d.get('regulation', '')}" for u, v, d in edge_list])
    else:
        edge_list = sorted(relabeled.edges(data=True))
        edge_str = ';'.join([f"{u}-{v}:{d.get('regulation', '')}" for u, v, d in edge_list])
    
    return edge_str  # Returns edge string including regulation

def count_motif_batch(subgraph_batch, network_type='directed'):
    """
    Count motifs in a batch of subgraphs.

    Parameters:
    - subgraph_batch (list of networkx.Graph): List of subgraphs to process.
    - network_type (str): 'directed' or 'undirected'.

    Returns:
    - dict: Counts of each canonical motif.
    """
    motif_counts = {}
    for sg in subgraph_batch:
        num_edges = sg.number_of_edges()
        if num_edges not in [2, 3]:
            logging.warning(f"Subgraph has unexpected number of edges: {num_edges}")
            continue
        cf = get_canonical_form(sg)
        motif_counts[cf] = motif_counts.get(cf, 0) + 1
    return motif_counts

def sample_connected_subgraphs_size_3(G, num_samples):
    """
    Sample connected 3-node subgraphs using a balanced random walk method to ensure diversity.

    Parameters:
    - G (networkx.Graph): The input graph.
    - num_samples (int): Number of connected subgraphs to sample.

    Yields:
    - networkx.Graph: Connected 3-node subgraphs.
    """
    logging.info(f"Sampling {num_samples} connected 3-node subgraphs using balanced random walk method...")
    sampled = 0
    nodes = list(G.nodes)
    
    with tqdm(total=num_samples, desc="Sampling Connected Subgraphs") as pbar:
        while sampled < num_samples:
            # Start with a random node
            node = random.choice(nodes)
            neighbors = list(G.successors(node)) if G.is_directed() else list(G.neighbors(node))
            if not neighbors:
                continue
            # Choose a random neighbor
            neighbor = random.choice(neighbors)
            
            # Decide randomly to choose a node connected to node or to neighbor
            if random.random() < 0.5:
                # Sample star-like subgraph
                connected_nodes = list(G.successors(node)) if G.is_directed() else list(G.neighbors(node))
            else:
                # Sample path-like subgraph
                connected_nodes = list(G.successors(neighbor)) if G.is_directed() else list(G.neighbors(neighbor))
            
            # Exclude the original node and neighbor to avoid duplication
            connected_nodes = list(set(connected_nodes) - {node, neighbor})
            if not connected_nodes:
                continue
            new_node = random.choice(connected_nodes)
            sub_nodes = [node, neighbor, new_node]
            subgraph = G.subgraph(sub_nodes).copy()
            # Verify connectedness
            if G.is_directed():
                if not nx.is_weakly_connected(subgraph):
                    continue
            else:
                if not nx.is_connected(subgraph):
                    continue
            yield subgraph
            sampled += 1
            pbar.update(1)

def random_sample_connected_subgraphs_size_3(G, num_samples):
    """
    Generate a list of randomly sampled connected 3-node subgraphs using an efficient method.

    Parameters:
    - G (networkx.Graph): The input graph.
    - num_samples (int): Number of random samples to generate.

    Returns:
    - list of networkx.Graph: List of sampled subgraphs.
    """
    return list(sample_connected_subgraphs_size_3(G, num_samples))

def get_regulation_distribution(G):
    """
    Calculate the distribution of regulation types in the graph.

    Parameters:
    - G (networkx.Graph or networkx.DiGraph): The input graph.

    Returns:
    - dict: Counts of each regulation type.
    """
    regulation_counts = {}
    for _, _, data in G.edges(data=True):
        regulation = data.get('regulation', '')
        regulation_counts[regulation] = regulation_counts.get(regulation, 0) + 1
    return regulation_counts

def custom_directed_double_edge_swap(G, nswap=10, max_tries=1000):
    """
    Perform a directed double edge swap to randomize the graph while preserving in-degree and out-degree sequences.

    Parameters:
    - G (networkx.DiGraph): The input directed graph.
    - nswap (int): Number of successful swaps to perform.
    - max_tries (int): Maximum number of attempts to perform swaps.

    Returns:
    - networkx.DiGraph: Randomized directed graph.
    """
    if not G.is_directed():
        raise nx.NetworkXError("Graph must be directed.")

    G_random = G.copy()
    tries = 0
    swaps = 0
    edges = list(G_random.edges())
    num_edges = len(edges)

    while swaps < nswap and tries < max_tries:
        tries += 1
        if num_edges < 2:
            break  # Not enough edges to swap
        # Randomly pick two distinct edges
        e1, e2 = random.sample(edges, 2)
        (a, b) = e1
        (c, d) = e2

        # Avoid self-loops and duplicate edges
        if a == d or c == b:
            continue
        if G_random.has_edge(a, d) or G_random.has_edge(c, b):
            continue

        # Perform the swap
        G_random.remove_edge(a, b)
        G_random.remove_edge(c, d)
        G_random.add_edge(a, d)
        G_random.add_edge(c, b)

        # Update the edges list
        edges.remove(e1)
        edges.remove(e2)
        edges.append((a, d))
        edges.append((c, b))

        swaps += 1

    if swaps < nswap:
        logging.warning(f"Only performed {swaps} out of {nswap} desired directed edge swaps.")

    return G_random

def randomize_network(G, num_swaps=10):
    """
    Randomize the network while preserving the degree sequence using double-edge swaps
    and reassign regulation types based on the original distribution.

    Parameters:
    - G (networkx.Graph or networkx.DiGraph): The input graph.
    - num_swaps (int): Number of swap attempts per edge.

    Returns:
    - networkx.Graph or networkx.DiGraph: Randomized graph with reassigned regulation types.
    """
    G_random = G.copy()
    try:
        if G_random.is_directed():
            logging.info("Randomizing network using custom directed double edge swap...")
            G_random = custom_directed_double_edge_swap(
                G_random,
                nswap=num_swaps * G_random.number_of_edges(),
                max_tries=num_swaps * G_random.number_of_edges() * 10
            )
        else:
            logging.info("Randomizing network using NetworkX's double_edge_swap...")
            total_swaps = num_swaps * G_random.number_of_edges()
            nx.double_edge_swap(
                G_random,
                nswap=total_swaps,
                max_tries=total_swaps * 10,
                seed=None  # Set a seed for reproducibility if desired
            )
        logging.info("Randomization complete.")

        # Reassign regulation types based on original distribution
        if G_random.is_directed():
            regulation_distribution = get_regulation_distribution(G)
            total_regulations = sum(regulation_distribution.values())
            regulation_prob = {k: v / total_regulations for k, v in regulation_distribution.items()}
            for u, v in G_random.edges():
                G_random[u][v]['regulation'] = random.choices(
                    population=list(regulation_prob.keys()),
                    weights=list(regulation_prob.values()),
                    k=1
                )[0]
    except nx.NetworkXError as e:
        logging.error(f"Randomization failed: {e}")
    return G_random

def compute_statistics(real_counts, random_counts_list):
    """
    Compute statistical significance of motifs with multiple testing correction.

    Parameters:
    - real_counts (dict): Counts of motifs in the real network.
    - random_counts_list (list of dict): Counts of motifs in randomized networks.

    Returns:
    - dict: Statistics including mean, std, z-score, and corrected p-value.
    """
    logging.info("Computing statistical significance of motifs...")
    stats_dict = {}
    p_values = []
    motifs = list(real_counts.keys())

    # Collect p-values for multiple testing correction
    for motif in motifs:
        real_count = real_counts[motif]
        random_counts = [rc.get(motif, 0) for rc in random_counts_list]
        mean_rand = np.mean(random_counts)
        std_rand = np.std(random_counts)
        z_score = (real_count - mean_rand) / std_rand if std_rand > 0 else 0
        p_value = np.mean([rc >= real_count for rc in random_counts])
        stats_dict[motif] = {
            'real': real_count,
            'mean_rand': mean_rand,
            'std_rand': std_rand,
            'z_score': z_score,
            'p_value': p_value
        }
        p_values.append(p_value)

    # Multiple testing correction using Benjamini-Hochberg
    corrected = multipletests(p_values, method='fdr_bh')
    for i, motif in enumerate(motifs):
        stats_dict[motif]['p_value_corrected'] = corrected[1][i]
        stats_dict[motif]['significant'] = corrected[0][i]

    logging.info("Statistical significance computation complete with multiple testing correction.")
    return stats_dict

def save_results(stats, size, network_name):
    """
    Save motif statistics to a CSV file.

    Parameters:
    - stats (dict): Motif statistics.
    - size (int): Size of motifs.
    - network_name (str): Name of the network.
    """
    df = pd.DataFrame.from_dict(stats, orient='index')
    df.index.name = 'motif'
    results_dir = './results'
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f'{network_name}_motif_stats_size_{size}.csv')
    df.to_csv(results_file)
    logging.info(f"Results saved to {results_file}")

def count_motifs_parallel(subgraphs, network_type='directed', num_workers=None):
    """
    Count motifs in parallel using multiprocessing.

    Parameters:
    - subgraphs (list of networkx.Graph): List of subgraphs to process.
    - network_type (str): 'directed' or 'undirected'.
    - num_workers (int): Number of parallel processes.

    Returns:
    - dict: Counts of each canonical motif.
    """
    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)  # Reserve one CPU

    logging.info(f"Starting parallel motif counting with {num_workers} workers...")
    motif_counts_total = {}

    # Split the subgraphs among workers
    chunks = [[] for _ in range(num_workers)]
    for i, sg in enumerate(subgraphs):
        chunks[i % num_workers].append(sg)

    with Pool(processes=num_workers) as pool:
        results = pool.map(partial(count_motif_batch, network_type=network_type), chunks)
        for motif_counts in results:
            for motif, count in motif_counts.items():
                motif_counts_total[motif] = motif_counts_total.get(motif, 0) + count

    logging.info("Parallel motif counting complete.")
    return motif_counts_total

def verify_degree_sequences(G_original, G_random):
    """
    Verify that the degree sequences match between the original and randomized networks.
    """
    if G_original.is_directed():
        orig_in_deg = sorted(dict(G_original.in_degree()).values())
        orig_out_deg = sorted(dict(G_original.out_degree()).values())
        rand_in_deg = sorted(dict(G_random.in_degree()).values())
        rand_out_deg = sorted(dict(G_random.out_degree()).values())
        if orig_in_deg != rand_in_deg or orig_out_deg != rand_out_deg:
            logging.warning("Degree sequences do not match after randomization.")
        else:
            logging.info("Degree sequences match between original and randomized networks.")
    else:
        orig_deg = sorted(dict(G_original.degree()).values())
        rand_deg = sorted(dict(G_random.degree()).values())
        if orig_deg != rand_deg:
            logging.warning("Degree sequences do not match after randomization.")
        else:
            logging.info("Degree sequences match between original and randomized networks.")

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Network Motif Detection Script')
    parser.add_argument('--network', type=str, required=True, help='Path to the network data file (.txt)')
    parser.add_argument('--network_name', type=str, required=True, help='Name of the network (e.g., E_coli, S_cerevisiae)')
    parser.add_argument('--directed', action='store_true', help='Flag indicating if the network is directed')
    parser.add_argument('--has_regulation', action='store_true', help='Flag indicating if the data includes regulation type (only for E. coli)')
    parser.add_argument('--size', type=int, default=3, choices=[3,4], help='Size of motifs to detect (3 or 4)')
    parser.add_argument('--num_random', type=int, default=100, help='Number of randomized networks to generate')
    parser.add_argument('--num_samples', type=int, default=100000, help='Number of random subgraphs to sample for motif counting')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel worker processes (default: all available cores minus one)')
    args = parser.parse_args()
    
    # Validate num_samples
    if args.num_samples <= 0:
        logging.error("Invalid value for --num_samples. It must be a positive integer.")
        return
    
    logging.info(f"Starting motif detection for network '{args.network_name}' with motif size {args.size}...")
    
    # Load the network
    G = load_network(args.network, network_type='directed' if args.directed else 'undirected', has_regulation=args.has_regulation)
    
    # Check connectivity and focus on the largest connected component
    if G.number_of_nodes() == 0:
        logging.error("The graph is empty. Exiting.")
        return
    
    if G.is_directed():
        if not nx.is_weakly_connected(G):
            logging.warning("The network is not weakly connected. Focusing on the largest weakly connected component.")
            largest_cc = max(nx.weakly_connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            logging.info(f"Largest weakly connected component has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    else:
        if not nx.is_connected(G):
            logging.warning("The network is not connected. Focusing on the largest connected component.")
            largest_cc = max(nx.connected_components(G), key=len)
            G = G.subgraph(largest_cc).copy()
            logging.info(f"Largest connected component has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Count motifs in the real network
    logging.info("Counting motifs in the real network...")
    if args.size == 3:
        real_subgraphs = random_sample_connected_subgraphs_size_3(G, args.num_samples)
    else:
        # Implement similar efficient sampling for size 4 if needed
        logging.error("Currently, only size=3 motifs are supported with the efficient sampling method.")
        return
    
    # Debugging: Log the first few sampled subgraphs
    for i, sg in enumerate(real_subgraphs[:10]):
        logging.info(f"Sampled subgraph {i}: {sg.edges(data=True)}")
    
    real_counts = count_motifs_parallel(real_subgraphs, 'directed' if G.is_directed() else 'undirected', num_workers=args.num_workers)
    logging.info(f"Unique motifs in real network: {len(real_counts)}")
    logging.info("Motif counting in real network complete.")
    
    # Generate randomized networks and count motifs
    random_counts_list = []
    logging.info(f"Generating and processing {args.num_random} randomized networks...")
    for i in range(1, args.num_random + 1):
        logging.info(f"Processing randomized network {i}/{args.num_random}...")
        randomized_G = randomize_network(G, num_swaps=10)  # Adjust num_swaps as needed
        verify_degree_sequences(G, randomized_G)
        
        # Check connectivity in randomized network
        if randomized_G.is_directed():
            if not nx.is_weakly_connected(randomized_G):
                logging.warning("Randomized network is not weakly connected. Focusing on the largest weakly connected component.")
                largest_cc_rand = max(nx.weakly_connected_components(randomized_G), key=len)
                randomized_G = randomized_G.subgraph(largest_cc_rand).copy()
                logging.info(f"Largest weakly connected component in randomized network {i} has {randomized_G.number_of_nodes()} nodes and {randomized_G.number_of_edges()} edges.")
        else:
            if not nx.is_connected(randomized_G):
                logging.warning("Randomized network is not connected. Focusing on the largest connected component.")
                largest_cc_rand = max(nx.connected_components(randomized_G), key=len)
                randomized_G = randomized_G.subgraph(largest_cc_rand).copy()
                logging.info(f"Largest connected component in randomized network {i} has {randomized_G.number_of_nodes()} nodes and {randomized_G.number_of_edges()} edges.")
        
        # Count motifs in the randomized network
        logging.info(f"Sampling {args.num_samples} connected 3-node subgraphs from randomized network {i}...")
        rand_subgraphs = random_sample_connected_subgraphs_size_3(randomized_G, args.num_samples)
        
        # Debugging: Log the first few randomized sampled subgraphs
        for j, sg in enumerate(rand_subgraphs[:2]):
            logging.info(f"Randomized {i} - Sampled subgraph {j}: {sg.edges(data=True)}")
        
        rand_counts = count_motifs_parallel(rand_subgraphs, 'directed' if randomized_G.is_directed() else 'undirected', num_workers=args.num_workers)
        random_counts_list.append(rand_counts)
        logging.info(f"Randomized network {i} complete.")
    
    # Compute statistics
    stats = compute_statistics(real_counts, random_counts_list)
    
    # Save results
    save_results(stats, args.size, args.network_name)
    
    logging.info("Motif detection analysis complete.")

if __name__ == "__main__":
    main()
