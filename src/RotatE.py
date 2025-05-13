from    .models  import BaseModel

import  torch
import numpy as np


class RotatE(BaseModel):
    def __init__(self, num_entities, num_relations, device, emb_dim=30, lr=1e-3):
        super(RotatE, self).__init__(num_entities, num_relations, emb_dim, device, lr)
        assert emb_dim % 2 == 0, "emb_dim must be even for RotatE (real + imaginary parts)"
        self.emb_dim = emb_dim
        self.complex_dim = emb_dim // 2
    
    def forward(self, pos_h, pos_r, pos_t):
        # Add normalization for RotatE
        self.entity_embds.data[:-1, :].div_(self.entity_embds.data[:-1, :].norm(p=2, dim=1, keepdim=True))
        h_embs = torch.index_select(self.entity_embds, 0, pos_h)
        t_embs = torch.index_select(self.entity_embds, 0, pos_t)
        r_embs = torch.index_select(self.rel_embds, 0, pos_r)
        return h_embs, r_embs, t_embs
    
    def predict(self, batch_h, batch_r, batch_t, batch_size, num_samples):
        h_embeds, r_embeds, t_embeds = self.forward(batch_h, batch_r, batch_t)
        
        h_re, h_im = h_embeds[:, :self.complex_dim], h_embeds[:, self.complex_dim:]
        t_re, t_im = t_embeds[:, :self.complex_dim], t_embeds[:, self.complex_dim:]
        r_re, r_im = r_embeds[:, :self.complex_dim], r_embeds[:, self.complex_dim:]
        
        r_norm = torch.sqrt(r_re**2 + r_im**2)
        r_re, r_im = r_re / r_norm, r_im / r_norm
        
        hr_re = h_re * r_re - h_im * r_im
        hr_im = h_re * r_im + h_im * r_re
        
        scores = torch.norm(hr_re - t_re, p=1, dim=1) + torch.norm(hr_im - t_im, p=1, dim=1)
        scores = scores.view(batch_size, num_samples).detach().cpu()
        return scores
    
    def compute_loss(self, batch):
        pos_h = batch[0].to(self.device)
        pos_r = batch[1].to(self.device)
        pos_t = batch[2].to(self.device)
        neg_h = batch[3].to(self.device)
        neg_r = batch[4].to(self.device)
        neg_t = batch[5].to(self.device)
        
        return self.RotatE_loss(pos_h, pos_r, pos_t, neg_h, neg_r, neg_t)
    
    def RotatE_loss(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_h_embs, pos_r_embs, pos_t_embs = self.forward(pos_h, pos_r, pos_t)
        neg_h_embs, neg_r_embs, neg_t_embs = self.forward(neg_h, neg_r, neg_t)
        
        pos_h_re, pos_h_im = pos_h_embs[:, :self.complex_dim], pos_h_embs[:, self.complex_dim:]
        pos_t_re, pos_t_im = pos_t_embs[:, :self.complex_dim], pos_t_embs[:, self.complex_dim:]
        pos_r_re, pos_r_im = pos_r_embs[:, :self.complex_dim], pos_r_embs[:, self.complex_dim:]
        
        neg_h_re, neg_h_im = neg_h_embs[:, :self.complex_dim], neg_h_embs[:, self.complex_dim:]
        neg_t_re, neg_t_im = neg_t_embs[:, :self.complex_dim], neg_t_embs[:, self.complex_dim:]
        neg_r_re, neg_r_im = neg_r_embs[:, :self.complex_dim], neg_r_embs[:, self.complex_dim:]
        
        pos_r_norm = torch.sqrt(pos_r_re**2 + pos_r_im**2)
        pos_r_re, pos_r_im = pos_r_re / pos_r_norm, pos_r_im / pos_r_norm
        neg_r_norm = torch.sqrt(neg_r_re**2 + neg_r_im**2)
        neg_r_re, neg_r_im = neg_r_re / neg_r_norm, neg_r_im / neg_r_norm
        
        pos_hr_re = pos_h_re * pos_r_re - pos_h_im * pos_r_im
        pos_hr_im = pos_h_re * pos_r_im + pos_h_im * pos_r_re
        neg_hr_re = neg_h_re * neg_r_re - neg_h_im * neg_r_im
        neg_hr_im = neg_h_re * neg_r_im + neg_h_im * neg_r_re
        
        d_pos = torch.norm(pos_hr_re - pos_t_re, p=1, dim=1) + torch.norm(pos_hr_im - pos_t_im, p=1, dim=1)
        d_neg = torch.norm(neg_hr_re - neg_t_re, p=1, dim=1) + torch.norm(neg_hr_im - neg_t_im, p=1, dim=1)
        
        ones = torch.ones(d_pos.size(0)).to(self.device)
        margin_loss = torch.nn.MarginRankingLoss(margin=1.)
        loss = margin_loss(d_neg, d_pos, ones)
        
        return loss
    
if __name__ == "__main__":
    num_entities = 1000
    num_relations = 50

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RotatE(num_entities, num_relations, device, emb_dim=30, lr=1e-3)
    train_losses = model._train(train_triples, train_batch_size=32, num_epochs=100)
    model._eval(eval_triples)