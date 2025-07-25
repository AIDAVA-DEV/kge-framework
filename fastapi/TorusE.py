import random

import torch
import numpy as np

from torch.utils.data import DataLoader


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
        
      return (torch.tensor(self.edges_index[idx][0], dtype=torch.long),
        self.edges_index[idx][1],
        torch.tensor(self.edges_index[idx][2], dtype=torch.long),
        neg_sample[0][0].detach().clone() if isinstance(neg_sample[0][0], torch.Tensor) else torch.tensor(neg_sample[0][0], dtype=torch.long),
        neg_sample[1].detach().clone() if isinstance(neg_sample[1], torch.Tensor) else torch.tensor(neg_sample[1], dtype=torch.long),
        neg_sample[0][1].detach().clone() if isinstance(neg_sample[0][1], torch.Tensor) else torch.tensor(neg_sample[0][1], dtype=torch.long))



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
    

class TorusE(BaseModel):
    def __init__(self, num_entities, num_relations, device, emb_dim=20, lr=1e-3):
        super(TorusE, self).__init__(num_entities, num_relations, emb_dim, device, lr)
        self.emb_dim = emb_dim
    
    def predict(self, batch_h, batch_r, batch_t, batch_size, num_samples):
        h_embeds, r_embeds, t_embeds = self.forward(batch_h, batch_r, batch_t)
        
        h_embeds = torch.sigmoid(h_embeds)
        r_embeds = torch.sigmoid(r_embeds)
        t_embeds = torch.sigmoid(t_embeds)
        
        diff = h_embeds + r_embeds - t_embeds
        dist = torch.min(torch.abs(diff), 1 - torch.abs(diff))
        
        scores = torch.sum(dist, dim=1)
        scores = scores.view(batch_size, num_samples).detach().cpu()
        return scores
    
    def compute_loss(self, batch):
        pos_h = batch[0].to(self.device)
        pos_r = batch[1].to(self.device)
        pos_t = batch[2].to(self.device)
        neg_h = batch[3].to(self.device)
        neg_r = batch[4].to(self.device)
        neg_t = batch[5].to(self.device)
        
        return self.TorusE_loss(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
    
    def TorusE_loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_h_embs, pos_r_embs, pos_t_embs = self.forward(pos_h, pos_r, pos_t)
        neg_h_embs, neg_r_embs, neg_t_embs = self.forward(neg_h, neg_r, neg_t)
        
        pos_h_embs = torch.sigmoid(pos_h_embs)
        pos_r_embs = torch.sigmoid(pos_r_embs)
        pos_t_embs = torch.sigmoid(pos_t_embs)
        neg_h_embs = torch.sigmoid(neg_h_embs)
        neg_r_embs = torch.sigmoid(neg_r_embs)
        neg_t_embs = torch.sigmoid(neg_t_embs)
        
        pos_diff = pos_h_embs + pos_r_embs - pos_t_embs
        pos_dist = torch.min(torch.abs(pos_diff), 1 - torch.abs(pos_diff))
        d_pos = torch.sum(pos_dist, dim=1)
        
        neg_diff = neg_h_embs + neg_r_embs - neg_t_embs
        neg_dist = torch.min(torch.abs(neg_diff), 1 - torch.abs(neg_diff))
        d_neg = torch.sum(neg_dist, dim=1)
        
        ones = torch.ones(d_pos.size(0)).to(self.device)
        margin_loss = torch.nn.MarginRankingLoss(margin=1.)
        loss = margin_loss(d_neg, d_pos, ones)
        
        return loss
    
    def get_entity_embeddings(self, entity_indices):
        """Retrieve embeddings for given entity indices."""
        return self.ent_embeddings(entity_indices)  # Assumes BaseModel has ent_embeddings

