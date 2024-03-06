from    models  import BaseModel
from    torch   import nn
import  torch

class TransE(nn.Module):
    
    def __init__(self,num_entities,num_relations,device,emb_dim=30, type_count=1,lr=1e-3):
        super(TransE,self).__init__()
        
        self.device = device
        
        self.num_entities = num_entities
        
        self.model        = BaseModel(num_entities,num_relations,emb_dim,device)

        params = list()
        params.append(self.model.entity_embds )
        params.append(self.model.rel_embds )
        
        """ ref: https://stackoverflow.com/questions/72679858/cant-optimize-a-non-leaf-tensor-on-torch-parameter """
        self.optimizer    = torch.optim.Adam(params,lr=lr)

    def predict(self,batch_h,batch_r ,batch_t,batch_size,num_samples):

        h_embeds,r_embeds,t_embeds = self.model(batch_h,batch_r,batch_t)
        scores = torch.norm(h_embeds+r_embeds-t_embeds,p=1,dim=1).view(batch_size,num_samples).detach().cpu()
        return scores

    
    def compute_loss(self,batch):
                     
        pos_h = batch[0].to(self.device)
        pos_r = batch[1].to(self.device)
        pos_t = batch[2].to(self.device)
        neg_h = batch[3].to(self.device)
        neg_r = batch[4].to(self.device)
        neg_t = batch[5].to(self.device)

        return self.TransE_loss(pos_h,pos_r,pos_t,neg_h,neg_r,neg_t)

    def TransE_loss(self,pos_h,pos_r,pos_t,neg_h,neg_r,neg_t):
        """ Ensures tail embeding and translated head embeding are nearest neighbour """
        
        pos_h_embs, pos_r_embs, pos_t_embs = self.model(pos_h ,pos_r ,pos_t)

        neg_h_embs, neg_r_embs, neg_t_embs = self.model(neg_h ,neg_r ,neg_t )
  
        d_pos = torch.norm(pos_h_embs + pos_r_embs - pos_t_embs, p=1, dim=1)
        d_neg = torch.norm(neg_h_embs + neg_r_embs - pos_t_embs, p=1, dim=1)
        ones  = torch.ones(d_pos.size(0)).to(self.device)

        margin_loss = torch.nn.MarginRankingLoss(margin=1.)
        loss        = margin_loss(d_neg,d_pos,ones)

        return loss
