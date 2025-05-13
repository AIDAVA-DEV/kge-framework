import  torch
from    torch import nn

from    torch.utils.data import Dataset
from    torch.utils.data import DataLoader

import  random
import  numpy as np

class TrainDataset(Dataset):

  def __init__(self,  edges   # list of index triples
               ,num_nodes     # number of entity nodes
               ,num_rels = 1  # number of links
               ,filter=True):

    self.edges_index  = edges    
    self.num_rels     = num_rels
    self.num_nodes    = num_nodes  
    self.num_edges    = len(edges)
    self.edges_dict   = {}
    self.filter       = filter

    
    # create a dict (for neg sampling)
    for i in range(self.num_edges):
      h = self.edges_index[i][0]
      t = self.edges_index[i][2]
      r = self.edges_index[i][1]
      ht = (h,t)
      if ht  not in self.edges_dict:
        self.edges_dict[ht] = []
      self.edges_dict[ht].append(r)

  def __len__(self):
      return self.num_edges
      
  def _sample_negative_edge(self,idx):

      sample  = random.uniform(0,1)
      found   = False

      while not found:
        if sample <= 0.4: # corrupt head
          h   = torch.randint(0,self.num_nodes,(1,))
          t   = self.edges_index[idx][2]
          r   = self.edges_index[idx][1]
        elif 0.4 < sample <= 0.8: # corrupt tail
          t   = torch.randint(0,self.num_nodes,(1,))
          h   = self.edges_index[idx][0]
          r   = self.edges_index[idx][1]
        else: # corrupt relation          
          r   = torch.randint(0,self.num_rels,(1,))[0]
          h   = self.edges_index[idx][0]
          t   = self.edges_index[idx][2]
        
        if not self.filter:
          found = True
        else:
          if (h,t) not in self.edges_dict:
            found = True
          elif r not in self.edges_dict[(h,t)]:
            found = True

      return [torch.tensor([h,t]),r]

  def __getitem__(self,idx):

      neg_sample  = self._sample_negative_edge(idx)        
        
      return torch.tensor(self.edges_index[idx][0]), self.edges_index[idx][1],   torch.tensor(self.edges_index[idx][2]), torch.tensor(neg_sample[0][0]) , torch.tensor(neg_sample[1]) , torch.tensor(neg_sample[0][1]) 


class TestDataset(Dataset):

  def __init__(self,  edges   # list of index triples
               ,num_nodes     # number of entity nodes
               ,num_rels = 1  # number of links
               ,filter = True
               ,mode = 'tail'): # for tail prediction

    self.edges_index  = edges    
    self.num_rels     = num_rels
    self.num_nodes    = num_nodes  
    self.num_edges    = len(edges)
    self.edges_dict   = {}
    self.filter       = filter
    self.mode         = mode
    
    # create a dict (for neg sample filtering)
    for i in range(self.num_edges):
      h = self.edges_index[i][0]
      t = self.edges_index[i][2]
      r = self.edges_index[i][1]
      if (h,t) not in self.edges_dict:
        self.edges_dict[(h,t)] = []
      self.edges_dict[(h,t)].append(r)

  def __len__(self):
      return self.num_edges

  def _sample_negative_edge(self,idx,max_num=100,mode='tail'):

      num_neg_samples = 0      
      triplets        = []      
      nodes           = list(range(self.num_nodes))
      random.shuffle(nodes)
      r               = self.edges_index[idx][1]

      while num_neg_samples < max_num:
                
        if mode == 'tail':
          t   = nodes[num_neg_samples]                 
          h   = self.edges_index[idx][0]
        else:
          t   = self.edges_index[idx][2]                  
          h   = nodes[num_neg_samples]                
        ht = torch.tensor([h,t]) 
                  
        if not self.filter:
          triplets.append([ht,r])
        else:
          if (h,t) not in self.edges_dict:
            triplets.append([ht,r])

          elif r not in self.edges_dict[(h,t)]:
            triplets.append([ht,r])

        num_neg_samples+=1
        if num_neg_samples == len(nodes):
          break

      return triplets

  def __getitem__(self,idx):

      pos_samples  = [torch.tensor([self.edges_index[idx][0],
                                    self.edges_index[idx][2]]),
                      self.edges_index[idx][1]]
    
      neg_samples  = self._sample_negative_edge(idx,mode=self.mode)      
        
      edges     = torch.stack([pos_samples[0]]+[ht for ht,_ in neg_samples])
      edge_rels = torch.stack([torch.tensor(pos_samples[1])] + [torch.tensor(r) for _,r in neg_samples]) 
        
      return edges, edge_rels


class BaseModel(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, emb_dim: int, device, lr=1e-3):
        super(BaseModel, self).__init__()
        self.num_entities = num_entities
        self.device = device
        self.entity_embds = nn.Parameter(torch.randn(num_entities, emb_dim))
        self.rel_embds = nn.Parameter(torch.randn(num_relations, emb_dim))
        params = [self.entity_embds, self.rel_embds]
        self.optimizer = torch.optim.Adam(params, lr=lr)
    
    def forward(self, pos_h, pos_r, pos_t):

        h_embs = torch.index_select(self.entity_embds, 0, pos_h)
        t_embs = torch.index_select(self.entity_embds, 0, pos_t)
        r_embs = torch.index_select(self.rel_embds, 0, pos_r)
        return h_embs, r_embs, t_embs

    def _train(self, train_triples, train_batch_size=32, num_epochs=100):
        train_db = TrainDataset(train_triples, self.num_entities, filter=False)
        train_dl = DataLoader(train_db, batch_size=train_batch_size, shuffle=True)
        
        log_freq = num_epochs // 10
        train_losses = []
        
        for e in range(num_epochs):
            self.train()
            losses = []
            
            for batch in train_dl:
                self.optimizer.zero_grad()
                loss = self.compute_loss(batch)
                loss.backward()
                self.optimizer.step()
                
                if np.isnan(loss.item()):
                    print('in _train: found invalid loss value, NaN')
                else:
                    losses.append(loss.item())
            
            if e % log_freq == 0:
                if len(losses) != 0:
                    mean_loss = np.array(losses).mean()
                    print(f'epoch {e},\t train loss {mean_loss:0.02f}')
                else:
                    mean_loss = np.NaN
                    print('in _train: found invalid mean loss, NaN')
                train_losses.append(mean_loss)
        
        return train_losses
    
    def _eval(self, eval_triples):
        self.eval()
        self.to(self.device)
        
        test_db = TestDataset(eval_triples, self.num_entities, filter=False)
        test_dl = DataLoader(test_db, batch_size=len(test_db), shuffle=False)
        
        batch = next(iter(test_dl))
        edges, edge_rels = batch
        batch_size, num_samples, _ = edges.size()
        edges = edges.view(batch_size * num_samples, -1)
        edge_rels = edge_rels.view(batch_size * num_samples, -1)
        
        h_indx = torch.tensor([int(x) for x in edges[:, 0]], device=self.device)
        r_indx = torch.tensor([int(x) for x in edge_rels.squeeze()], device=self.device)
        t_indx = torch.tensor([int(x) for x in edges[:, 1]], device=self.device)
        
        scores = self.predict(h_indx, r_indx, t_indx, batch_size, num_samples)
        
        argsort = torch.argsort(scores, dim=1, descending=False)
        rank_list = torch.nonzero(argsort == 0, as_tuple=False)
        rank_list = rank_list[:, 1] + 1
        
        hits1_list = [(rank_list <= 1).to(torch.float).mean()]
        hits10_list = [(rank_list <= 10).to(torch.float).mean()]
        MR_list = [rank_list.to(torch.float).mean()]
        MRR_list = [(1. / rank_list.to(torch.float)).mean()]
        
        hits1 = sum(hits1_list) / len(hits1_list)
        hits10 = sum(hits10_list) / len(hits10_list)
        mr = sum(MR_list) / len(MR_list)
        mrr = sum(MRR_list) / len(MRR_list)
        
        print(f'hits@1 {hits1.item():0.02f} hits@10 {hits10.item():0.02f} MR {mr.item():0.02f} MRR {mrr.item():0.02f}')

