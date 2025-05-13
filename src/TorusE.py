from    .models  import BaseModel

import  torch
import  numpy as np

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
    model = TorusE(num_entities, num_relations, device, emb_dim=emb_dim, lr=lr)
    
    # Train the model
    print("Training TorusE model...")
    train_losses = model._train(train_triples, train_batch_size=train_batch_size, num_epochs=num_epochs)
    
    # Evaluate the model
    print("\nEvaluating TorusE model...")
    model._eval(eval_triples)