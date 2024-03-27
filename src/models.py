import  torch
from    torch import nn

class BaseModel(nn.Module):
    
    def __init__(self,
                num_entities    : int, # total number of node types 
                num_relations   : int, # total number of link types
                emb_dim         : int,  # size of embedding vectors
            ):  
        
        super(BaseModel,self).__init__()
        
        # initialize embeddings with random normal values
        self.entity_embds = nn.Parameter(torch.randn(num_entities,emb_dim))
        self.rel_embds    = nn.Parameter(torch.randn(num_relations,emb_dim))

            
            
    def forward(self,pos_h  # head node indexes
                ,pos_r      # link indexes
                ,pos_t      # trail node indexes     
            ): 
        
        """ Normalize """
        self.entity_embds.data[:-1,:].div_(self.entity_embds.data[:-1,:].norm(p=2,dim=1,keepdim=True))
        
        
        """ Extract Embeddings """
        h_embs = torch.index_select(self.entity_embds,0,pos_h)
        t_embs = torch.index_select(self.entity_embds,0,pos_t)
        r_embs = torch.index_select(self.rel_embds   ,0,pos_r) 
        
        return h_embs, r_embs, t_embs
    