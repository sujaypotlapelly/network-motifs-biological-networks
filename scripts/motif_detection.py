import networkx as nx
import itertools
import random
import os
import numpy as np
from scipy import stats
import pandas as pd
import argparse
from multiprocessing import Pool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if has_regulation:
                if len(parts) < 3:
                    continue  # Skip malformed lines
                regulator, target, regulation = parts[:3]
                G.add_edge(regulator, target, regulation=regulation)
            else:
                if len(parts) < 2:
                    continue  # Skip malformed lines
                regulator, target = parts[:2]
                G.add_edge(regulator, target)
    
    return G

def enumerate_subgraphs(G, size):
    """
    Enumerate all connected subgraphs of a given size.
    
    Parameters:
    - G (networkx.Graph): The input graph.
    - size (int): Number of nodes in the subgraph.
    
    Returns:
    - List of subgraphs (networkx.Graph).
    """
    subgraphs = []
    for nodes in itertools.combinations(G.nodes, size):
        sg = G.subgraph(nodes).copy()
        if G.is_directed():
            if sg.number_of_edges() == 0:
                continue  # Skip disconnected or empty subgraphs
        else:
            if not nx.is_connected(sg):
                continue  # Skip disconnected subgraphs
        subgraphs.append(sg)
    return subgraphs

def get_canonical_form(subgraph):
    """
    Get a canonical string representation of a subgraph using graph6 format.
    
    Parameters:
    - subgraph (networkx.Graph): The subgraph.
    
    Returns:
    - str: Canonical string representation.
    """
    # Sort nodes to ensure consistent ordering
    sorted_nodes = sorted(subgraph.nodes())
    mapping = {node: i for i, node in enumerate(sorted_nodes)}
    sg_relabel = nx.relabel_nodes(subgraph, mapping)
    return nx.to_graph6_bytes(sg_relabel, nodes=sorted_nodes).decode().strip()

def count_motifs(G, size):
    """
    Count motifs of a specific size in the graph.
    
    Parameters:
    - G (networkx.Graph): The input graph.
    - size (int): Number of nodes in the motif.
    
    Returns:
    - dict: Counts of each canonical motif.
    """
    motif_counts = {}
    subgraphs = enumerate_subgraphs(G, size)
    for sg in subgraphs:
        cf = get_canonical_form(sg)
        motif_counts[cf] = motif_counts.get(cf, 0) + 1
    return motif_counts

def randomize_network(G, num_swaps=10):
    """
    Randomize the network while preserving the degree sequence using double-edge swaps.
    
    Parameters:
    - G (networkx.Graph): The input graph.
    - num_swaps (int): Number of swap attempts.
    
    Returns:
    - networkx.Graph: Randomized graph.
    """
    G_random = G.copy()
    try:
        if G_random.is_directed():
            nx.double_edge_swap(G_random, nswap=num_swaps * G_random.number_of_edges(), max_tries=num_swaps * G_random.number_of_edges() * 10)
        else:
            nx.double_edge_swap(G_random, nswap=num_swaps * G_random.number_of_edges(), max_tries=num_swaps * G_random.number_of_edges() * 10)
    except nx.NetworkXError as e:
        logging.error(f"Randomization failed: {e}")
    return G_random

def compute_statistics(real_counts, random_counts_list):
    """
    Compute statistical significance of motifs.
    
    Parameters:
    - real_counts (dict): Counts of motifs in the real network.
    - random_counts_list (list of dict): Counts of motifs in randomized networks.
    
    Returns:
    - dict: Statistics including mean, std, z-score, and p-value.
    """
    stats_dict = {}
    for motif, count in real_counts.items():
        random_counts = [rc.get(motif, 0) for rc in random_counts_list]
        mean = np.mean(random_counts)
        std = np.std(random_counts)
        z_score = (count - mean) / std if std > 0 else 0
        p_value = np.mean([rc >= count for rc in random_counts])
        stats_dict[motif] = {'real': count, 'mean_rand': mean, 'std_rand': std, 'z_score': z_score, 'p_value': p_value}
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
    results_dir = '../results'
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f'{network_name}_motif_stats_size_{size}.csv')
    df.to_csv(results_file)
    logging.info(f"Results saved to {results_file}")

def process_network(args):
    """
    Process a single network: count motifs in real and randomized networks and compute statistics.
    
    Parameters:
    - args: Parsed command-line arguments.
    """
    # Load the network
    G = load_network(args.network, network_type='directed' if args.directed else 'undirected', has_regulation=args.has_regulation)
    logging.info(f"Loaded network '{args.network_name}' with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    # Count motifs in the real network
    logging.info(f"Counting {args.size}-node motifs in the real network...")
    real_counts = count_motifs(G, args.size)
    logging.info(f"Real network motif counts: {real_counts}")
    
    # Generate randomized networks and count motifs
    random_counts_list = []
    logging.info(f"Generating {args.num_random} randomized networks and counting motifs...")
    for i in range(args.num_random):
        G_rand = randomize_network(G)
        rand_counts = count_motifs(G_rand, args.size)
        random_counts_list.append(rand_counts)
        if (i+1) % 10 == 0 or (i+1) == args.num_random:
            logging.info(f"Processed {i+1}/{args.num_random} randomized networks.")
    
    # Compute statistics
    logging.info("Computing statistical significance of motifs...")
    stats = compute_statistics(real_counts, random_counts_list)
    
    # Save results
    save_results(stats, args.size, args.network_name)

def main():
    parser = argparse.ArgumentParser(description='Network Motif Detection Script')
    parser.add_argument('--network', type=str, required=True, help='Path to the network data file (.txt)')
    parser.add_argument('--network_name', type=str, required=True, help='Name of the network (e.g., E_coli, S_cerevisiae)')
    parser.add_argument('--directed', action='store_true', help='Flag indicating if the network is directed')
    parser.add_argument('--has_regulation', action='store_true', help='Flag indicating if the data includes regulation type (only for E. coli)')
    parser.add_argument('--size', type=int, default=3, choices=[3,4], help='Size of motifs to detect (3 or 4)')
    parser.add_argument('--num_random', type=int, default=100, help='Number of randomized networks to generate')
    args = parser.parse_args()
    
    process_network(args)

if __name__ == "__main__":
    main()
