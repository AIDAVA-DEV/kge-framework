import  pandas as pd
import  numpy as np
import  math
from    collections import defaultdict
import  torch
from    torch_geometric.data import HeteroData
import  rdflib
import  operator
import  random


### GNN HELPER FUNCIONS ### 

def eval_hits(edge_index, tail_pred, output, max_num):    
    top1 = 0
    top10 = 0
    n = edge_index.size(1)

    for idx in range(n):
        if tail_pred == 1:
            x = torch.index_select(output, 0, edge_index[0, idx])
        else:
            x = torch.index_select(output, 0, edge_index[1, idx])
        
        candidates, candidates_embeds = sample_negative_edges_idx(idx=idx,
                                                                  edge_index=edge_index,
                                                                  tail_pred=tail_pred,
                                                                  output=output,
                                                                  max_num=max_num)

        distances = torch.cdist(candidates_embeds, x, p=2)
        dist_dict = {cand: dist for cand, dist in zip(candidates, distances)} 

        sorted_dict = dict(sorted(dist_dict.items(), key=operator.itemgetter(1), reverse=True))
        sorted_keys = list(sorted_dict.keys())

        ranks_dict = {sorted_keys[i]: i for i in range(0, len(sorted_keys))}
        rank = ranks_dict[edge_index[1, idx].item()]
        
        if rank <= 1:
            top1 += 1
        if rank <= 10:
            top10 += 1
    return top1/n, top10/n

def sample_negative_edges_idx(idx, edge_index, tail_pred, output, max_num):
    num_neg_samples = 0
    candidates = []
    nodes = list(range(edge_index.size(1)))
    random.shuffle(nodes)

    while num_neg_samples < max_num:    
        if tail_pred == 1:
            t = nodes[num_neg_samples]
            h = edge_index[0, idx].item()
            if h not in edge_index[0] or t not in edge_index[1]:
                candidates.append(t)
        else: 
            t = edge_index[1, idx].item()
            h = nodes[num_neg_samples]
            if h not in edge_index[0] or t not in edge_index[1]:
                candidates.append(h)
        num_neg_samples += 1
    candidates_embeds = torch.index_select(output, 0, torch.tensor(candidates))

    if tail_pred == 1:
        true_tail = edge_index[1, idx]
        candidates.append(true_tail.item())
        candidates_embeds = torch.concat([candidates_embeds, torch.index_select(output, 0, true_tail)])
    else:
        true_head = edge_index[0, idx]
        candidates.append(true_head.item())
        candidates_embeds = torch.concat([candidates_embeds, torch.index_select(output, 0, true_head)])
    return candidates, candidates_embeds

def get_data(g, nodes_dict, relations_dict):
    edge_data = defaultdict(list)
    for s, p, o in g.triples((None, None, None)):
        src, dst, rel = nodes_dict[s], nodes_dict[o], relations_dict[p]
        edge_data['edge_index'].append([src, dst])
        edge_data['edge_type'].append(rel)
    
    data = HeteroData(edge_index=torch.tensor(edge_data['edge_index'], dtype=torch.long).t().contiguous(),
                      edge_type=torch.tensor(edge_data['edge_type'], dtype=torch.long))
    
    return data

def split_edges(data, test_ratio = 0.2, val_ratio = 0):    
    row, col = data.edge_index
    edge_type = data.edge_type

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    perm = torch.randperm(row.size(0))
    row, col, edge_type = row[perm], col[perm], edge_type[perm]

    r, c, e = row[:n_v], col[:n_v], edge_type[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    data.val_edge_type = e
    r, c, e = row[n_v:n_v + n_t], col[n_v:n_v + n_t], edge_type[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    data.test_edge_type = e
    r, c, e = row[n_v + n_t:], col[n_v + n_t:], edge_type[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    data.train_edge_type = e
    
    return data

def copy_graph(g):
    new_g = rdflib.Graph()

    for triple in g:
        new_g.add(triple)
    
    return new_g