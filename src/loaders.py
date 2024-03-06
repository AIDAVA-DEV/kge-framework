from    torch.utils.data import Dataset
import  torch 
import  random

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
