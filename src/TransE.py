from   .models  import BaseModel
from    torch   import nn
import  torch


import  numpy as np

class TransE(BaseModel):
    def __init__(self, num_entities, num_relations, device, emb_dim=30, lr=1e-3):
        super(TransE, self).__init__(num_entities, num_relations, emb_dim, device, lr)
        self.emb_dim = emb_dim
    
    def forward(self, pos_h, pos_r, pos_t):
        # Add normalization for TransE
        self.entity_embds.data[:-1, :].div_(self.entity_embds.data[:-1, :].norm(p=2, dim=1, keepdim=True))
        h_embs = torch.index_select(self.entity_embds, 0, pos_h)
        t_embs = torch.index_select(self.entity_embds, 0, pos_t)
        r_embs = torch.index_select(self.rel_embds, 0, pos_r)
        return h_embs, r_embs, t_embs
    
    def predict(self, batch_h, batch_r, batch_t, batch_size, num_samples):
        h_embeds, r_embeds, t_embeds = self.forward(batch_h, batch_r, batch_t)
        scores = torch.norm(h_embeds + r_embeds - t_embeds, p=1, dim=1).view(batch_size, num_samples).detach().cpu()
        return scores
    
    def compute_loss(self, batch):
        pos_h = batch[0].to(self.device)
        pos_r = batch[1].to(self.device)
        pos_t = batch[2].to(self.device)
        neg_h = batch[3].to(self.device)
        neg_r = batch[4].to(self.device)
        neg_t = batch[5].to(self.device)
        
        return self.TransE_loss(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
    
    def TransE_loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_h_embs, pos_r_embs, pos_t_embs = self.forward(pos_h, pos_r, pos_t)
        neg_h_embs, neg_r_embs, neg_t_embs = self.forward(neg_h, neg_r, neg_t)
        
        d_pos = torch.norm(pos_h_embs + pos_r_embs - pos_t_embs, p=1, dim=1)
        d_neg = torch.norm(neg_h_embs + neg_r_embs - neg_t_embs, p=1, dim=1)
        ones = torch.ones(d_pos.size(0)).to(self.device)
        
        margin_loss = torch.nn.MarginRankingLoss(margin=1.)
        loss = margin_loss(d_neg, d_pos, ones)
        
        return loss
    
    

if __name__ == "__main__":
    # Small example parameters
    num_entities = 10
    num_relations = 5
    emb_dim = 20
    lr = 1e-3
    train_batch_size = 16
    num_epochs = 10
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate synthetic triples
    np.random.seed(42)
    train_triples = [
        (np.random.randint(0, num_entities),
         np.random.randint(0, num_relations),
         np.random.randint(0, num_entities))
        for _ in range(100)
    ]
    eval_triples = [
        (np.random.randint(0, num_entities),
         np.random.randint(0, num_relations),
         np.random.randint(0, num_entities))
        for _ in range(20)
    ]
    
    # Initialize model
    model = TransE(num_entities, num_relations, device, emb_dim=emb_dim, lr=lr)
    
    # Train the model
    print("Training TransE model...")
    train_losses = model._train(train_triples, train_batch_size=train_batch_size, num_epochs=num_epochs)
    
    # Evaluate the model
    print("\nEvaluating TransE model...")
    model._eval(eval_triples)