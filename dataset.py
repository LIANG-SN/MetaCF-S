import os
import numpy as np
from torch.utils.data import Dataset
import random
import json
import copy
import collections

class MetaTrainData(Dataset):
    def __init__(self, args, split):
        super(MetaTrainData, self).__init__()
        self.n_users = args.n_users
        self.n_items = args.n_items
        self.data_root = args.data_root
        self.test_k_shot = args.test_k_shot
        self.split = split
        self.h = args.h
        self.topk = args.topk
        self.use_score = args.use_score
        self.user_item_list, self.eval_user_item_list = collections.defaultdict(list), {}
        self.user_score_list = {}
        self.user_full_item_list = {}
        self.user_list, self.total_user_list = [], []
        self.item_list = list(range(self.n_users, self.n_users+self.n_items))
        self.interaction_matrix = np.zeros((self.n_users, self.n_items))
        self.score_matrix = np.zeros((self.n_users, self.n_items))
        self.original_model = args.original_model
        self.soft_matrix = args.soft_matrix
        if args.dataset == 'kindle':
            train_file = self.data_root+"train.txt"
            test_file = self.data_root+"test.txt"
            with open(train_file) as f:
                for line in f:
                    line = line.strip().split()
                    user = int(line[0])
                    self.user_list.append(user)
                    self.total_user_list.append(user)
                    items = [int(item) for item in line[1:]]
                    self.user_item_list[user] = items
                    for item in items:
                        self.interaction_matrix[user, item-self.n_users] = 1
                        self.user_item_list[item].append(user)
            if split=="test":
                self.user_list = []
                with open(test_file) as f:
                    for line in f:
                        line = line.strip().split()
                        user = int(line[0])
                        self.user_list.append(user)
                        self.total_user_list.append(user)
                        items = [int(item) for item in line[1:]]
                        self.eval_user_item_list[user] = items[5:]
                        items = items[:self.test_k_shot]
                        self.user_item_list[user] = items
                        for item in items:
                            self.interaction_matrix[user, item-self.n_users] = 1
                            self.user_item_list[item].append(user)
        elif args.dataset == 'movie':
            train_x_file = os.path.join(self.data_root, 'warm_state.json')
            train_y_file = os.path.join(self.data_root, 'warm_state_y.json')
            test_x_file = os.path.join(self.data_root, 'user_cold_state.json')
            test_y_file = os.path.join(self.data_root, 'user_cold_state_y.json')

            with open(train_x_file, encoding="utf-8") as f:
                train_x_struct = json.loads(f.read())
            with open(train_y_file, encoding="utf-8") as f:
                train_y_struct = json.loads(f.read())
            with open(test_x_file, encoding="utf-8") as f:
                test_x_struct = json.loads(f.read())
            with open(test_y_file, encoding="utf-8") as f:
                test_y_struct = json.loads(f.read())

            for _, user_id in enumerate(train_x_struct.keys()):
                items_raw = [int(item)+self.n_users for item in train_x_struct[user_id]]
                items = []
                scores = []
                full_items = []
                for item, y in zip(items_raw, train_y_struct[user_id]):
                    y = int(y)
                    full_items.append(item)
                    scores.append(y)
                    if y >= 3:
                        items.append(item)
                    
                if len(items) < 13 or len(items) > 100:
                    continue

                user_id = int(user_id)
                self.user_list.append(user_id)
                self.total_user_list.append(user_id)
                self.user_item_list[user_id] = items
                self.user_score_list[user_id] = scores
                self.user_full_item_list[user_id] = full_items

                for item, score in zip(full_items, scores):
                    self.score_matrix[user_id, item-self.n_users] = score

                for item in items:
                    self.interaction_matrix[user_id, item-self.n_users] = 1
                    self.user_item_list[item].append(user_id)
            if split=="test":
                self.user_list = []
                for _, user_id in enumerate(test_x_struct.keys()):
                    items_raw = [int(item)+self.n_users for item in test_x_struct[user_id]]
                    items = []
                    scores = []
                    full_items = []
                    for item, y in zip(items_raw, test_y_struct[user_id]):
                        y = int(y)
                        full_items.append(item)
                        scores.append(y)
                        if y >= 3:
                            items.append(item)
                    
                    if len(items) < 13 or len(items) > 100:
                        continue

                    
                    user_id = int(user_id)
                    self.user_list.append(user_id)
                    self.total_user_list.append(user_id)
                    # self.eval_user_item_list[user_id] = items[-10:] 
                    # self.user_item_list[user_id] = items[:-10]
                    if self.use_score:
                        # todo: shuffle
                        items_q = full_items[-10:]
                        items_s = full_items[:-10]
                        items_interact = list(set(items) & set(items_s))

                    else:
                        items_q = items[self.test_k_shot:]
                        items_interact = items[:self.test_k_shot]
                    
                    self.eval_user_item_list[user_id] = items_q
                    self.user_item_list[user_id] = items_interact
                    self.user_full_item_list[user_id] = full_items
                    self.user_score_list[user_id] = scores
                    
                    for item, score in zip(full_items, scores):
                        self.score_matrix[user_id, item-self.n_users] = score
                    for item in items_interact:
                        self.interaction_matrix[user_id, item-self.n_users] = 1
                        self.user_item_list[item].append(user_id)
        
    def __getitem__(self, index):
        if self.use_score:
            return self.get_item_score(index)
        else:
            return self.get_item_binary(index)
    
    def get_item_score(self, index):
        user = self.user_list[index]
        items_full = self.user_full_item_list[user]
        items_interact = self.user_item_list[user]
        if self.soft_matrix:
            cur_interaction_matrix = self.score_matrix / 5.0
            cur_interaction_matrix[cur_interaction_matrix<0.45] = 0
        else:
            cur_interaction_matrix = self.interaction_matrix
        cur_user_item_list = copy.deepcopy(self.user_item_list)

        if self.split=="train":
            k = random.randint(1,self.test_k_shot)
            k = min(k, len(items_full))
            try:
                support_items = random.sample(items_full, k)
                # if self.soft_matrix:
                #     support_items = support_items + list(random.sample(self.item_list, 1))
            except:
                print('sample support fail', index, user, len(items_full), k)
                exit()
            query_items = random.sample(items_full, k) 
            
            # items_full_rand = copy.deepcopy(items_full)
            # random.shuffle(items_full_rand)
            # support_items = items_full_rand[:-10]
            # if len(support_items) > 10:
            #     support_items = support_items[:10]
            # query_items = items_full_rand[-10:]

            cur_user_item_list[user] = support_items
            for item in list(set(items_interact)-set(support_items)):
                cur_user_item_list[item].remove(user)
        else:
            # support_items = items_full[:-10]
            # if len(support_items) > 10:
            #     support_items = support_items[:10]
            query_items = items_full[-10:]
            support_items = list(set(items_full)-set(query_items))
            # if self.soft_matrix:
            #     support_items = support_items + list(random.sample(self.item_list, 1))
            k = min(self.test_k_shot, len(support_items))
            support_items = random.sample(support_items, k)



        max_node = 0
        support_x, support_subgraph, support_index = [], [], []
        for item in support_items:
            x, subgraph, index = self.sample_subgraph(user, item, cur_interaction_matrix, cur_user_item_list, support_items)
            max_node = max(max_node, len(x))
            support_x.append(x)
            support_subgraph.append(subgraph)
            support_index.append(index)
        support_x, support_subgraph = self.pad(support_x, support_subgraph, max_node)
        query_x, query_subgraph, query_index = [], [], []
        max_node = 0
        for item in query_items:
            x, subgraph, index = self.sample_subgraph(user, item, cur_interaction_matrix, cur_user_item_list, support_items)
            max_node = max(max_node, len(x))
            query_x.append(x)
            query_subgraph.append(subgraph)
            query_index.append(index)
        query_x, query_subgraph = self.pad(query_x, query_subgraph, max_node)
        # get y
        support_items_index = [items_full.index(it) for it in support_items]
        query_items_index = [items_full.index(it) for it in query_items]
        support_y = [self.user_score_list[user][idx] for idx in support_items_index]
        query_y = [self.user_score_list[user][idx] for idx in query_items_index]

        return support_x, support_subgraph, support_index, support_y, \
                query_x, query_subgraph, query_index, query_y, user
    
            

    def get_item_binary(self, index):
        user = self.user_list[index]
        items = self.user_item_list[user]
        cur_interaction_matrix = self.interaction_matrix
        cur_user_item_list = copy.deepcopy(self.user_item_list)
        if self.split=="train":
            k = random.randint(1,self.test_k_shot)
            k = min(k, len(items))
            try:
                support_items = random.sample(items, k)
            except:
                print('sample support fail', index, user, len(items), k)
                exit()
            query_items = random.sample(items, k)
            negative_samples = list(set(self.item_list)-set(items))
            support_neg_items = random.sample(negative_samples, k)
            query_neg_items = random.sample(negative_samples, k)
            cur_user_item_list[user] = support_items
            for item in list(set(items)-set(support_items)):
                cur_user_item_list[item].remove(user)
            max_node = 0
            support_x, support_subgraph, support_index = [], [], []
            for item in support_items:
                x, subgraph, index = self.sample_subgraph(user, item, cur_interaction_matrix, cur_user_item_list, support_items)
                max_node = max(max_node, len(x))
                support_x.append(x)
                support_subgraph.append(subgraph)
                support_index.append(index)
            support_x, support_subgraph = self.pad(support_x, support_subgraph, max_node)
            query_x, query_subgraph, query_index = [], [], []
            max_node = 0
            for item in query_items:
                x, subgraph, index = self.sample_subgraph(user, item, cur_interaction_matrix, cur_user_item_list, support_items)
                max_node = max(max_node, len(x))
                query_x.append(x)
                query_subgraph.append(subgraph)
                query_index.append(index)
            query_x, query_subgraph = self.pad(query_x, query_subgraph, max_node)
            support_neg_x, support_neg_subgraph, support_neg_index = [], [], []
            max_node = 0
            for item in support_neg_items:
                x, subgraph, index = self.sample_subgraph(user, item, cur_interaction_matrix, cur_user_item_list, support_items)
                max_node = max(max_node, len(x))
                support_neg_x.append(x)
                support_neg_subgraph.append(subgraph)
                support_neg_index.append(index)
            support_neg_x, support_neg_subgraph = self.pad(support_neg_x, support_neg_subgraph, max_node)
            query_neg_x, query_neg_subgraph, query_neg_index = [], [], []
            max_node = 0
            for item in query_neg_items:
                x, subgraph, index = self.sample_subgraph(user, item, cur_interaction_matrix, cur_user_item_list, support_items)
                max_node = max(max_node, len(x))
                query_neg_x.append(x)
                query_neg_subgraph.append(subgraph)
                query_neg_index.append(index)
            query_neg_x, query_neg_subgraph = self.pad(query_neg_x, query_neg_subgraph, max_node)
            return support_x, support_subgraph, support_index, support_neg_x, support_neg_subgraph, support_neg_index, \
                    query_x, query_subgraph, query_index, query_neg_x, query_neg_subgraph, query_neg_index
        else:
            k = self.test_k_shot
            support_items = items
            negative_samples = list(set(self.item_list)-set(support_items))
            support_neg_items = random.sample(negative_samples, k)
            test_pos_items = self.eval_user_item_list[user][:self.topk]
            test_negative_item = list(set(self.item_list)-set(support_items)-set(test_pos_items))
            test_neg_items = random.sample(test_negative_item, self.topk*10-1)
            max_node = 0
            support_x, support_subgraph, support_index = [], [], []
            for item in support_items:
                x, subgraph, index = self.sample_subgraph(user, item, cur_interaction_matrix, cur_user_item_list, support_items)
                max_node = max(max_node, len(x))
                support_x.append(x)
                support_subgraph.append(subgraph)
                support_index.append(index)
            support_x, support_subgraph = self.pad(support_x, support_subgraph, max_node)
            support_neg_x, support_neg_subgraph, support_neg_index = [], [], []
            max_node = 0
            for item in support_neg_items:
                x, subgraph, index = self.sample_subgraph(user, item, cur_interaction_matrix, cur_user_item_list, support_items)
                max_node = max(max_node, len(x))
                support_neg_x.append(x)
                support_neg_subgraph.append(subgraph)
                support_neg_index.append(index)
            support_neg_x, support_neg_subgraph = self.pad(support_neg_x, support_neg_subgraph, max_node)
            max_node = 0
            test_x, test_subgraph, test_index = [], [], []
            for item in test_pos_items:
                x, subgraph, index = self.sample_subgraph(user, item, cur_interaction_matrix, cur_user_item_list, support_items)
                max_node = max(max_node, len(x))
                test_x.append(x)
                test_subgraph.append(subgraph)
                test_index.append(index)
            for item in test_neg_items:
                x, subgraph, index = self.sample_subgraph(user, item, cur_interaction_matrix, cur_user_item_list, support_items)
                max_node = max(max_node, len(x))
                test_x.append(x)
                test_subgraph.append(subgraph)
                test_index.append(index) # index: n_user nodes sampled
            test_x, test_subgraph = self.pad(test_x, test_subgraph, max_node)
            return support_x, support_subgraph, support_index, support_neg_x, support_neg_subgraph, support_neg_index, user, \
            test_x, test_subgraph, test_index
           

    def pad(self, xs, subgraphs, max_node):
        out_xs, out_subgraphs = [], []
        for i in range(len(xs)):
            x, subgraph = xs[i], subgraphs[i]
            cur_len = len(x)
            out_x = np.zeros((max_node))
            out_subgraph = np.zeros((max_node, max_node))
            out_subgraph[:cur_len, :cur_len] = subgraph
            out_x[:cur_len] = x
            out_xs.append(out_x)
            out_subgraphs.append(out_subgraph)
        return np.array(out_xs), np.array(out_subgraphs)


    def sample_subgraph(self, user, item, cur_interaction_matrix, cur_user_item_list, support_items):
        u_nodes = [user]
        v_nodes = [item]
        u_dist, v_dist = [0], [0]
        u_visited, v_visited = set([user]), set([item])
        u_fringe, v_fringe = set([user]), set([item])
        for dist in range(1, self.h+1):
            v_fringe, u_fringe = self.prob_neighbors(u_fringe, cur_user_item_list), \
                    self.prob_neighbors(v_fringe, cur_user_item_list)
            u_fringe = u_fringe - u_visited
            v_fringe = v_fringe - v_visited
            u_visited = u_visited.union(u_fringe)
            v_visited = v_visited.union(v_fringe)
            u_nodes = u_nodes + list(u_fringe)
            v_nodes = v_nodes + list(v_fringe)
            u_dist = u_dist + [dist] * len(u_fringe)
            v_dist = v_dist + [dist] * len(v_fringe)
        subgraph = cur_interaction_matrix[u_nodes][:, np.array(v_nodes)-self.n_users]
        subgraph[0] = 0
        for support_item in support_items:
            if support_item in v_nodes: # todo: check this
                support_item_index = v_nodes.index(support_item)
                # subgraph[0, support_item_index] = cur_interaction_matrix[user, support_item]
                subgraph[0, support_item_index] = 1
        subgraph[0,0] = 0 
        out_x = [x*2 for x in u_dist] + [x*2+1 for x in v_dist]
        n_users, n_items = len(u_nodes), len(v_nodes)
        item_index = n_users
        out_subgraph = np.eye(n_users+n_items)
        out_subgraph[:n_users, n_users:] = subgraph
        out_subgraph[n_users:, :n_users] = subgraph.transpose()
        sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(out_subgraph, axis=0, dtype=float)))
        out_subgraph = np.matmul(np.matmul(sqrt_deg, out_subgraph), sqrt_deg)
        return out_x, out_subgraph, item_index

    def neighbors(self, objs, cur_user_item_list):
        out_objs = []
        for obj in objs:
            out_objs.extend(cur_user_item_list[obj])
        return set(out_objs)

    def prob_neighbors(self, objs, cur_user_item_list):
        out_objs = []
        for obj in objs:
            for nb in cur_user_item_list[obj]:
                if obj <= self.n_users:
                    s = self.score_matrix[obj, nb-self.n_users]
                else:
                    s = self.score_matrix[nb, obj-self.n_users]
                if s <= 2:
                    p = 0
                elif s == 3:
                    p = 0.5
                elif s == 4:
                    p = 0.8
                elif s == 5:
                    p = 1
                
                if self.original_model:
                    p=1 
                
                if np.random.binomial(1, p):
                    out_objs.append(nb)
            if self.soft_matrix:
                if obj <= self.n_users:
                    out_objs.append(random.sample(self.item_list, 1)[0])
        return set(out_objs)

    def __len__(self):
        return len(self.user_list)



