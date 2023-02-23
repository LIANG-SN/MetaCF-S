import torch
from torch.nn import functional as F
import torch.nn as nn

class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__(in_features, out_features)
        self.weight.fast = None 
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.linear(x, self.weight.fast, self.bias.fast) 
        else:
            out = super(Linear, self).forward(x)
        return out

class Embedding(nn.Embedding): 
    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__(num_embeddings, embedding_dim)
        self.weight.fast = None 

    def forward(self, x):
        if self.weight.fast is not None:
            out = F.embedding(x, self.weight.fast)
        else:
            out = super(Embedding, self).forward(x)
        return out

class Meta(torch.nn.Module):
    def __init__(self, args):
        super(Meta, self).__init__()
        self.embed_dim = args.embed_dim
        self.fc_dim = args.hidden_dim
        self.dropout = args.drop
        self.lr_inner_init = args.lr_inner_init
        self.wd = args.wd
        self.n_users = args.n_users
        self.n_items = args.n_items
        self.embed = Embedding(2*(args.h+1), self.embed_dim)
        self.fc1 = Linear(self.embed_dim, self.fc_dim)
        self.fc2 = Linear(self.fc_dim, self.fc_dim)
        if args.embed_train:
            self.local_part = nn.ModuleList([self.embed, self.fc1, self.fc2])
            self.define_task_lr()
            self.global_part = nn.ModuleList([self.embed, self.fc1, self.fc2, self.task_lr])
        else:
            self.local_part = nn.ModuleList([self.fc1, self.fc2])
            self.define_task_lr()
            self.global_part = nn.ModuleList([self.fc1, self.fc2, self.task_lr])

    def define_task_lr(self):
        self.task_lr = nn.ParameterList()
        for k, parameter in enumerate(self.local_part.parameters()):
            self.task_lr.append(nn.Parameter(self.lr_inner_init * torch.ones_like(parameter, requires_grad=True)))
            
    
    def forward(self, x_pos, A_pos, index_pos, x_neg, A_neg, index_neg):
        K = len(index_pos)
        x_pos = self.embed(x_pos)
        x_neg = self.embed(x_neg)
        x_pos = F.dropout(F.relu(self.fc1(torch.bmm(A_pos, x_pos))), p=self.dropout)
        x_pos = self.fc2(torch.bmm(A_pos, x_pos))
        x_neg = F.dropout(F.relu(self.fc1(torch.bmm(A_neg, x_neg))), p=self.dropout)
        x_neg = self.fc2(torch.bmm(A_neg, x_neg))
        range_k = list(range(K))
        u_pos = x_pos[:,0,:]
        i_pos = x_pos[range_k, index_pos]
        u_neg = x_neg[:,0,:]
        i_neg = x_neg[range_k, index_neg]
        pos_scores = torch.sum(u_pos*i_pos, dim=1)
        neg_scores = torch.sum(u_neg*i_neg, dim=1)
        loss = torch.mean(-F.logsigmoid(pos_scores-neg_scores))
        if self.wd:
            reg = torch.norm(u_pos, p=2) + torch.norm(i_pos, p=2) + torch.norm(u_neg, p=2) + torch.norm(i_neg, p=2)
            loss += self.wd*reg/K
        return loss

    def run_test(self, x, A, index):
        K = len(index)
        x = self.embed(x)
        x = F.dropout(F.relu(self.fc1(torch.bmm(A, x))), p=self.dropout)
        x = self.fc2(torch.bmm(A, x))
        range_k = list(range(K))
        u = x[:,0,:]
        i = x[range_k, index]
        # print(u.shape, i.shape, len(index))
        # exit()
        # 100
        scores = torch.sum(u*i, dim=1)
        return scores


class Meta_score(torch.nn.Module):
    def __init__(self, args):
        super(Meta_score, self).__init__()
        self.embed_dim = args.embed_dim
        self.fc_dim = args.hidden_dim
        self.dropout = args.drop
        self.lr_inner_init = args.lr_inner_init
        self.wd = args.wd
        self.n_users = args.n_users
        self.n_items = args.n_items
        self.dot_prod = args.dot_prod
        self.embed = Embedding(2*(args.h+1), self.embed_dim)
        self.fc1 = Linear(self.embed_dim, self.fc_dim)
        self.fc2 = Linear(self.fc_dim, self.fc_dim)
        if self.dot_prod:
            if args.embed_train:
                self.local_part = nn.ModuleList([self.embed, self.fc1, self.fc2])
                self.define_task_lr()
                self.global_part = nn.ModuleList([self.embed, self.fc1, self.fc2, self.task_lr])
            else:
                self.local_part = nn.ModuleList([self.fc1, self.fc2])
                self.define_task_lr()
                self.global_part = nn.ModuleList([self.fc1, self.fc2, self.task_lr])
        else:
            self.fc3 = Linear(self.fc_dim * 2, 1)
            if args.embed_train:
                self.local_part = nn.ModuleList([self.embed, self.fc1, self.fc2, self.fc3])
                self.define_task_lr()
                self.global_part = nn.ModuleList([self.embed, self.fc1, self.fc2, self.fc3, self.task_lr])
            else:
                self.local_part = nn.ModuleList([self.fc1, self.fc2, self.fc3])
                self.define_task_lr()
                self.global_part = nn.ModuleList([self.fc1, self.fc2, self.fc3, self.task_lr])
        

    def define_task_lr(self):
        self.task_lr = nn.ParameterList()
        for k, parameter in enumerate(self.local_part.parameters()):
            self.task_lr.append(nn.Parameter(self.lr_inner_init * torch.ones_like(parameter, requires_grad=True)))
            
    
    def forward(self, x_pos, A_pos, index_pos):
        # A: (B, N+M, N+M)
        # x: # (B, N+M, D)
        K = len(index_pos) # K=B
        x_pos = self.embed(x_pos) # (B, N+M, D)
        x_pos = F.dropout(F.relu(self.fc1(torch.bmm(A_pos, x_pos))), p=self.dropout)
        x_pos = self.fc2(torch.bmm(A_pos, x_pos))
        range_k = list(range(K))
        u_pos = x_pos[:,0,:] # (B, D)
        i_pos = x_pos[range_k, index_pos] # (B, D)
        
        if self.dot_prod:
            pos_scores = torch.sum(u_pos*i_pos, dim=1)
        else:
            pos_scores = self.fc3(torch.cat([u_pos, i_pos], -1))[:, 0]
        
        return pos_scores

    def run_test(self, x, A, index):
        K = len(index)
        x = self.embed(x)
        x = F.dropout(F.relu(self.fc1(torch.bmm(A, x))), p=self.dropout)
        x = self.fc2(torch.bmm(A, x))
        range_k = list(range(K))
        u = x[:,0,:]
        i = x[range_k, index]
        # print(u.shape, i.shape, len(index))
        # exit()
        # 100
        if self.dot_prod:
            scores = torch.sum(u*i, dim=1)
        else:
            scores = self.fc3(torch.cat([u, i], -1))[:, 0]
        
        return scores
