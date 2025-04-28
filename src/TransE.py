from    src.models  import BaseModel as bm
from    torch   import nn
import  torch

from    src.archive.loaders import TestDataset, TrainDataset
from    torch.utils.data import DataLoader

import  numpy as np

class TransE(nn.Module):
    
    def __init__(self,num_entities,num_relations,device,emb_dim=30,lr=1e-3):
        super(TransE,self).__init__()
        
        self.device = device
        
        self.num_entities = num_entities
        
        self.model        = bm(num_entities,num_relations,emb_dim).to(device)

        params = list()
        params.append(self.model.entity_embds )
        params.append(self.model.rel_embds )
        
        """ ref: https://stackoverflow.com/questions/72679858/cant-optimize-a-non-leaf-tensor-on-torch-parameter """
        
        self.optimizer    = torch.optim.Adam(params,lr=lr)

        # initialize embeddings with random normal values
        self.entity_embds = nn.Parameter(torch.randn(num_entities,emb_dim))
        self.rel_embds    = nn.Parameter(torch.randn(num_relations,emb_dim))
    
    def _train(self , train_triples,
               train_batch_size = 32 ,
               num_epochs      = 100):

        train_db = TrainDataset(train_triples,self.num_entities,filter=False)
        train_dl = DataLoader(train_db,batch_size=train_batch_size,shuffle=True)
        

        log_freq = num_epochs//10
        
        train_losses = []
        for e in range(num_epochs):  

            self.model.train()
            losses = []

            for batch in train_dl:

                self.optimizer.zero_grad()

                # late binding to specific model via compute_loss
                loss = self.compute_loss(batch)

                loss.backward()
                self.optimizer.step()
                
                if np.isnan(loss.item()):
                    print('in _train: found invalid loss value, NaN')
                else:
                    losses.append(loss.item())

            if e % log_freq == 0:
                if len(losses)!=0:
                    mean_loss = np.array(losses).mean()
                    print('epoch {},\t train loss {:0.02f}'.format(e,mean_loss))
                else:
                    mean_loss = np.NaN
                    print('in _train: found invalid mean loss, NaN')
                
            train_losses.append(mean_loss)

        return train_losses
    
    def _eval(self, eval_triples):
        
        self.model.eval() 
        self.model.to(self.device)

        test_db = TestDataset(eval_triples,self.num_entities,filter=False)
        test_dl = DataLoader(test_db,batch_size=len(test_db),shuffle=False)

        # load all
        batch = next(iter(test_dl))
        
        edges, edge_rels            = batch
        batch_size, num_samples, _  = edges.size()
        edges                       = edges.view(batch_size*num_samples,-1)
        edge_rels                   = edge_rels.view(batch_size*num_samples,-1)

        h_indx = torch.tensor([int(x) for x in edges[:,0]],device=self.device)
        r_indx = torch.tensor([int(x) for x in edge_rels.squeeze()],device=self.device)
        t_indx = torch.tensor([int(x) for x in edges[:,1]],device=self.device)
        
        scores = self.predict(h_indx,r_indx,t_indx,batch_size,num_samples)

        # sort and calculate scores
        argsort   = torch.argsort(scores,dim = 1,descending= False)
        rank_list = torch.nonzero(argsort==0,as_tuple=False)
        rank_list = rank_list[:,1] + 1
        
        hits1_list  = []
        hits10_list = []
        MR_list     = []
        MRR_list    = []
  
        hits1_list.append( (rank_list <= 1).to(torch.float).mean() )
        hits10_list.append( (rank_list <= 10).to(torch.float).mean() )
        MR_list.append(rank_list.to(torch.float).mean())
        MRR_list.append( (1./rank_list.to(torch.float)).mean() )
  
        hits1   = sum(hits1_list)/len(hits1_list)
        hits10  = sum(hits10_list)/len(hits10_list)
        mr      = sum(MR_list)/len(MR_list)
        mrr     = sum(MRR_list)/len(MRR_list)

        print(f'hits@1 {hits1.item():0.02f} hits@10 {hits10.item():0.02f}',
              f' MR {mr.item():0.02f} MRR  {mrr.item():0.02f}')
    


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
