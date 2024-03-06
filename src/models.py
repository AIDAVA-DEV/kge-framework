import  torch
from    torch import nn
import  numpy as np

from    src.loaders import DataLoader, TestDataset, TrainDataset

class BaseModel(nn.Module):
    
    def __init__(self,
                num_entities    : int, # total number of node types 
                num_relations   : int, # total number of link types
                emb_dim         : int,  # size of embedding vectors
                device  # cpu or gpu  
            ):  
        
        super(BaseModel,self).__init__()
        
        # initialize embeddings with random normal values
        self.entity_embds = nn.Parameter(torch.randn(num_entities,emb_dim))
        self.rel_embds    = nn.Parameter(torch.randn(num_relations,emb_dim))

        self.device       = device
            
            
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
    
    def _train(self , train_triples,
               train_batch_size = 32 ,
               num_epochs      = 100):

        train_db = TrainDataset(train_triples,self.num_entities,filter=False)
        train_dl = DataLoader(train_db,batch_size=train_batch_size,shuffle=True)

        self.model.to(self.device)

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
        
        scores = self.model.predict(h_indx,r_indx,t_indx,batch_size,num_samples)

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
    
