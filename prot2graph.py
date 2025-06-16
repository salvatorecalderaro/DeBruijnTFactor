import torch
from torch_geometric.nn.encoding import PositionalEncoding
import networkx as nx
from torch_geometric.utils import from_networkx

aminoacidi = 'ACDEFGHIKLMNPQRSTVWYX'
EMB_SIZE = 256

def check_input(sequence):
    # Standard amino acids
    standard_aas = set(aminoacidi)

    # Normalize sequence
    sequence = sequence.upper()
    sequence = sequence.replace(" ", " ").replace("\n", "").replace("\r", "")

    # Replace non-standard AAs with 'X'
    cleaned_sequence = ''.join([aa if aa in standard_aas else 'X' for aa in sequence])
    print(cleaned_sequence)
    return cleaned_sequence



def one_hot_encoding(sequence):
    aa_to_index = {aa: i for i, aa in enumerate(aminoacidi)}
    one_hot = torch.zeros((len(sequence), len(aminoacidi)))
    for i, aa in enumerate(sequence):
        one_hot[i, aa_to_index[aa]] = 1

    one_hot=one_hot.T.flatten()
    return one_hot

def generate_all_kmers(sequence,k):
    kmers=[sequence[i:i+k] for i in range(len(sequence)-k+1)]
    return kmers



def create_positional_encoding_kmers(kmers, d):
    kmers_pos_enc = {}
    kmers_count = {}
    positional_encoding = PositionalEncoding(d)

    for i, kmer in enumerate(kmers):
        pe = positional_encoding(torch.tensor([[i]])).squeeze(0).tolist()

        if kmer in kmers_pos_enc:
            kmers_pos_enc[kmer] = [sum(x) for x in zip(kmers_pos_enc[kmer], pe)]
            kmers_count[kmer] += 1
        else:
            kmers_pos_enc[kmer] = pe
            kmers_count[kmer] = 1

    for kmer in kmers_pos_enc:
        kmers_pos_enc[kmer] = [x / kmers_count[kmer] for x in kmers_pos_enc[kmer]]

    return kmers_pos_enc

def create_dbg(sequence, k, emb_size):
    k_1_mers = generate_all_kmers(sequence, k - 1)
    d = emb_size - (len(aminoacidi) * (k - 1))
    pos_enc = create_positional_encoding_kmers(k_1_mers, d)
    g = nx.Graph()
    feats = []
    for kmer in generate_all_kmers(sequence, k):
        prefix = kmer[:-1]
        suffix = kmer[1:]
        g.add_edge(prefix, suffix)
    for node in g.nodes():
        ohe = one_hot_encoding(node)
        pe = torch.tensor(pos_enc[node])
        feats.append(torch.concatenate([ohe, pe]))

    feats = torch.stack(feats)

    G = from_networkx(g)
    G.x = feats
    return G

def create_graphs(sequence):
    k_sizes = [3, 5, 7, 9, 11, 13]
    graphs = []
    for k in k_sizes:
        graphs.append(create_dbg(sequence, k, EMB_SIZE))
    return graphs
    


