import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing
import copy
import logging
import argparse
import heapq
import itertools
from sklearn.metrics import ndcg_score
from dataset import *
from model import Meta, Meta_score

torch.multiprocessing.set_sharing_strategy('file_system')
def parse_args():
    parser = argparse.ArgumentParser([],description='')

    parser.add_argument('--cuda', type=int, default=1, help='cuda device')
    parser.add_argument('--tasks_per_metaupdate', type=int, default=32, help='number of tasks in each batch per meta-update')
    parser.add_argument('--lr_inner_init', type=float, default=0.01, help='inner-loop learning rate (per task)')
    parser.add_argument('--lr_meta', type=float, default=1e-3, help='outer-loop learning rate (used with Adam optimiser)')
    parser.add_argument('--h', type=int, default=2, help='hop neighbors, we typically use two layer GCN')
    parser.add_argument('--drop', type=float, default=0, help='Dropout ratio')
    parser.add_argument('--wd', type=float, default=0, help='Weight Decay')
    parser.add_argument('--save_interval', type=int, default=1, help='saving interval')
    parser.add_argument('--num_grad_steps_inner', type=int, default=5, help='number of gradient steps in inner loop (during training)')
    parser.add_argument('--num_grad_steps_eval', type=int, default=5, help='number of gradient updates at test time (for evaluation)')
    parser.add_argument('--test_k_shot', type=int, default=3, help='number of finetuned samples when evaluation')
    parser.add_argument('--topk', type=int, default=1, help='top 10 ranking')
    parser.add_argument('--data_root', type=str, default="./dataset/kindle/", help='path to data root')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--embed_train', action='store_false', default=True, help='whether not to optimize the initial embedding')
    parser.add_argument('--embed_dim', type=int, default=64, help='embedding_dim')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden_dim')
    parser.add_argument('--num_epoch', type=int, default=3, help='num of epoches to use')
    parser.add_argument('--eval', action='store_true', default=False, help='evaluation mode')
    parser.add_argument('--dataset', type=str, default='movie', help='dataset type')
    parser.add_argument('--save_dir', type=str, default='save_models', help='')
    parser.add_argument('--soft_matrix', action='store_true', default=False)
    parser.add_argument('--use_score', action='store_true', default=False, help='use score')
    parser.add_argument('--test_warm', action='store_true', default=False)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--dot_prod', action='store_true', default=False)
    parser.add_argument('--original_model', action='store_true', default=False)

    args = parser.parse_args()
    return args

def train(args, save_path='save_models'):
    model = Meta(args).cuda()
    meta_optimiser = torch.optim.Adam(model.parameters(), args.lr_meta)
    meta_grad_init = [0 for _ in range(len(model.global_part.state_dict()))]
    train_dataset = DataLoader(MetaTrainData(args, split="train"), batch_size=1, num_workers=args.num_workers)
    test_dataset = DataLoader(MetaTrainData(args, split="test"), batch_size=1, num_workers=args.num_workers)
    for epoch in range(args.num_epoch):
        model.train()
        support_x, support_subgraph, support_index, support_neg_x, support_neg_subgraph, support_neg_index, \
                    query_x, query_subgraph, query_index, query_neg_x, query_neg_subgraph, query_neg_index = [],[],[],[],[],[],[],[],[],[],[],[]
        iter_counter = 0
        task_num = 0
        meta_grad = copy.deepcopy(meta_grad_init)
        loss_pre = []
        loss_after = []
        for _, batch in enumerate(train_dataset):
            support_x = torch.tensor(batch[0][0]).long().cuda()
            support_subgraph = torch.tensor(batch[1][0]).to(torch.float32).cuda()
            support_index = batch[2]
            support_neg_x = torch.tensor(batch[3][0]).long().cuda()
            support_neg_subgraph = torch.tensor(batch[4][0]).to(torch.float32).cuda()
            support_neg_index = batch[5]
            query_x = torch.tensor(batch[6][0]).long().cuda()
            query_subgraph = torch.tensor(batch[7][0]).to(torch.float32).cuda()
            query_index = batch[8]
            query_neg_x = torch.tensor(batch[9][0]).long().cuda()
            query_neg_subgraph = torch.tensor(batch[10][0]).to(torch.float32).cuda()
            query_neg_index = batch[11]
            fast_parameters = model.local_part.parameters()
            for weight in model.local_part.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_inner):
                loss = model(support_x, support_subgraph, support_index, support_neg_x, support_neg_subgraph, support_neg_index)
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                fast_parameters = []
                for k, weight in enumerate(model.local_part.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - model.task_lr[k] * grad[k]
                    else:
                        weight.fast = weight.fast - model.task_lr[k] * grad[k]  
                    fast_parameters.append(weight.fast)         

            loss_q = model(query_x, query_subgraph, query_index, query_neg_x, query_neg_subgraph, query_neg_index)
            loss_after.append(loss_q.item())
            task_grad_test = torch.autograd.grad(loss_q, model.global_part.parameters())
            
            for g in range(len(task_grad_test)):
                meta_grad[g] += task_grad_test[g].detach()
            task_num+=1
            if task_num==args.tasks_per_metaupdate:
                meta_optimiser.zero_grad()

                for c, param in enumerate(model.global_part.parameters()):
                    param.grad = meta_grad[c] / float(args.tasks_per_metaupdate)
                    param.grad.data.clamp_(-10, 10)

                meta_optimiser.step()
                support_x, support_subgraph, support_index, support_neg_x, support_neg_subgraph, support_neg_index, \
                        query_x, query_subgraph, query_index, query_neg_x, query_neg_subgraph, query_neg_index = [],[],[],[],[],[],[],[],[],[],[],[]
                
                loss_pre = np.array(loss_pre)
                loss_after = np.array(loss_after)
                print('epoch {} Iter {} - - loss: {}'.format(epoch, iter_counter, np.mean(loss_after)))
                logging.info('epoch {} Iter {} - - loss: {}'.format(epoch, iter_counter, np.mean(loss_after)))
                iter_counter += 1
                meta_grad = copy.deepcopy(meta_grad_init)
                loss_pre = []
                loss_after = []
                task_num = 0
        if epoch % (args.save_interval) == 0:
            torch.save(model.state_dict(), os.path.join(save_path,"epoch_{}.pt".format(epoch)))
            hr, ndcg = evaluate_test(args, model, test_dataset)
            print('epoch {} Test Set - - HR: {}, NDCG: {}'.format(epoch, hr, ndcg))
            logging.info('epoch {} Test Set - - HR: {}, NDCG: {}'.format(epoch, hr, ndcg))

    return model

def train_score(args, save_path='save_models'):
    model = Meta_score(args).cuda()

    if args.start_epoch > 0:
        ckpt_path = os.path.join(args.save_dir, 'epoch_{}.pt'.format(args.start_epoch-1))
        model.load_state_dict(torch.load(ckpt_path))
    meta_optimiser = torch.optim.Adam(model.parameters(), args.lr_meta)
    meta_grad_init = [0 for _ in range(len(model.global_part.state_dict()))]
    train_dataset = DataLoader(MetaTrainData(args, split="train"), batch_size=1, num_workers=args.num_workers)
    test_dataset = DataLoader(MetaTrainData(args, split="test"), batch_size=1, num_workers=args.num_workers)
    for epoch in range(args.start_epoch, args.num_epoch):
        model.train()
        support_x, support_subgraph, support_index, \
            query_x, query_subgraph, query_index = [],[],[],[],[],[]
        iter_counter = 0
        task_num = 0
        meta_grad = copy.deepcopy(meta_grad_init)
        loss_after = []
        for _, batch in enumerate(train_dataset):
            support_x = torch.tensor(batch[0][0]).long().cuda()
            support_subgraph = torch.tensor(batch[1][0]).to(torch.float32).cuda()
            support_index = batch[2][0]
            support_y = torch.tensor(batch[3], dtype=torch.float32).cuda()
            query_x = torch.tensor(batch[4][0]).long().cuda()
            query_subgraph = torch.tensor(batch[5][0]).to(torch.float32).cuda()
            query_index = batch[6][0]
            query_y = torch.tensor(batch[7], dtype=torch.float32).cuda()

            if not args.original_model:
                support_index = batch[2]
                query_index = batch[6]

            fast_parameters = model.local_part.parameters()
            for weight in model.local_part.parameters():
                weight.fast = None
            for k in range(args.num_grad_steps_inner):
                scores_s = model(support_x, support_subgraph, support_index)
                loss = torch.mean((scores_s - support_y)**2)
                grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
                fast_parameters = []
                for k, weight in enumerate(model.local_part.parameters()):
                    if weight.fast is None:
                        weight.fast = weight - model.task_lr[k] * grad[k]
                    else:
                        weight.fast = weight.fast - model.task_lr[k] * grad[k]  
                    fast_parameters.append(weight.fast)

            scores_q = model(query_x, query_subgraph, query_index)
            loss_q = torch.mean((scores_q - query_y)**2)
            loss_after.append(loss_q.item())
            task_grad_test = torch.autograd.grad(loss_q, model.global_part.parameters())
            
            for g in range(len(task_grad_test)):
                meta_grad[g] += task_grad_test[g].detach()
            task_num+=1
            if task_num==args.tasks_per_metaupdate:
                meta_optimiser.zero_grad()

                for c, param in enumerate(model.global_part.parameters()):
                    param.grad = meta_grad[c] / float(args.tasks_per_metaupdate)
                    param.grad.data.clamp_(-10, 10)
                meta_optimiser.step()
                support_x, support_subgraph, support_index, \
                    query_x, query_subgraph, query_index = [],[],[],[],[],[]
                loss_after = np.array(loss_after)
                print('epoch {} Iter {} - loss: {}'.format(epoch, iter_counter, np.mean(loss_after)))
                logging.info('epoch {} Iter {} - loss: {}'.format(epoch, iter_counter, np.mean(loss_after)))
                iter_counter += 1
                meta_grad = copy.deepcopy(meta_grad_init)
                loss_after = []
                task_num = 0
        if epoch % (args.save_interval) == 0:
            torch.save(model.state_dict(), os.path.join(save_path,"epoch_{}.pt".format(epoch)))
            ndcg1, ndcg3, mae = evaluate_test_score(args, model, test_dataset)
            print('epoch {} Test Set - NDCG1: {}, NDCG3: {}, MAE: {}'.format(
                epoch, ndcg1, ndcg3, mae))
            logging.info('epoch {} Test Set - NDCG1: {}, NDCG3: {}, MAE: {}'.format(
                epoch, ndcg1, ndcg3, mae))
    return model

def argmax_top_k(a, top_k=10):
    ele_idx = heapq.nlargest(top_k, zip(a, itertools.count()))
    return np.array([idx for ele, idx in ele_idx], dtype=np.intc)

def ndcg(rank, ground_truth):
    len_rank = len(rank)
    len_gt = len(ground_truth)
    idcg_len = min(len_gt, len_rank)

    idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
    idcg[idcg_len:] = idcg[idcg_len-1]
    dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
    result = dcg/idcg
    return result[-1]

def hr(rank, ground_truth):
    if len(set(rank)&set(ground_truth)):
        return 1
    else:
        return 0

def evaluate_test(args, model, dataloader):
    model.eval()
    ndcg1_list, ndcg3_list, hr_list = [], [], []
    for _, batch in enumerate(dataloader):
        support_x = torch.tensor(batch[0][0]).long().cuda()
        support_subgraph = torch.tensor(batch[1][0]).to(torch.float32).cuda()
        support_index = batch[2]
        support_neg_x = torch.tensor(batch[3][0]).long().cuda()
        support_neg_subgraph = torch.tensor(batch[4][0]).to(torch.float32).cuda()
        support_neg_index = batch[5][0]
        user = batch[6][0].item()
        test_x = torch.tensor(batch[7][0]).long().cuda()
        test_subgraph = torch.tensor(batch[8][0]).to(torch.float32).cuda()
        test_index = batch[9]

        fast_parameters = model.local_part.parameters()
        for weight in model.local_part.parameters():
            weight.fast = None
        for k in range(args.num_grad_steps_eval):
            loss = model(support_x, support_subgraph, support_index, support_neg_x, support_neg_subgraph, support_neg_index)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []
            for k, weight in enumerate(model.local_part.parameters()):
                if weight.fast is None:
                    weight.fast = weight - model.task_lr[k] * grad[k]
                else:
                    weight.fast = weight.fast - model.task_lr[k] * grad[k]  
                fast_parameters.append(weight.fast)
        user_index = user
        preds = model.run_test(test_x, test_subgraph, test_index)
        preds = preds.detach().cpu().numpy()
        gt1 = [0]
        gt = list(range(args.topk))
        ranking = argmax_top_k(preds, args.topk)
        ndcg1_user = ndcg(ranking, gt1)
        ndcg3_user = ndcg(ranking, gt)
        hr_user = hr(ranking, gt)
        print('user_id: {}, hr: {}, ndcg1: {}, ndcg3: {}'.format(
            user_index, hr_user, ndcg1_user, ndcg3_user))
        logging.info('user_id: {}, hr: {}, ndcg1: {}, ndcg3: {}'.format(
            user_index, hr_user, ndcg1_user, ndcg3_user))
        ndcg1_list.append(ndcg1_user)
        ndcg3_list.append(ndcg3_user)
        hr_list.append(hr_user)
    return np.mean(hr_list), np.mean(ndcg1_list), np.mean(ndcg3_list)

def evaluate_test_score(args, model, dataloader):
    model.eval()
    ndcg1_list, ndcg3_list, mae_list = [], [], []
    for _, batch in enumerate(dataloader):
        support_x = torch.tensor(batch[0][0]).long().cuda()
        support_subgraph = torch.tensor(batch[1][0]).to(torch.float32).cuda()
        support_index = batch[2][0]
        support_y = torch.tensor(batch[3], dtype=torch.float32).cuda()
        test_x = torch.tensor(batch[4][0]).long().cuda()
        test_subgraph = torch.tensor(batch[5][0]).to(torch.float32).cuda()
        test_index = batch[6][0]
        test_y = np.array(batch[7], dtype=np.float32)
        user = batch[8][0].item()

        if not args.original_model:
            support_index = batch[2]
            test_index = batch[6]

        fast_parameters = model.local_part.parameters()
        for weight in model.local_part.parameters():
            weight.fast = None
        for k in range(args.num_grad_steps_eval):
            score = model(support_x, support_subgraph, support_index)
            loss = torch.mean((score-support_y)**2)
            grad = torch.autograd.grad(loss, fast_parameters, create_graph=True)
            fast_parameters = []
            for k, weight in enumerate(model.local_part.parameters()):
                if weight.fast is None:
                    weight.fast = weight - model.task_lr[k] * grad[k]
                else:
                    weight.fast = weight.fast - model.task_lr[k] * grad[k]  
                fast_parameters.append(weight.fast)
        user_index = user
        preds = model.run_test(test_x, test_subgraph, test_index)
        preds = preds.detach().cpu().numpy()

        test_y = test_y[None, :]
        preds = preds[None, :]

        mae = np.mean(np.abs(preds-test_y))
        ndcg1 = ndcg_score(test_y, preds, k=1)
        ndcg3 = ndcg_score(test_y, preds, k=3)
        print('user_id: {}, ndcg1: {}, ndcg3: {}, mae: {}, gt: {}, pred: {}'.format(
            user_index, ndcg1, ndcg3, mae, test_y[0], preds[0]))
        logging.info('user_id: {}, ndcg1: {}, ndcg3: {}, mae: {}, gt: {}, pred: {}'.format(
            user_index, ndcg1, ndcg3, mae, test_y[0], preds[0]))
        ndcg1_list.append(ndcg1)
        ndcg3_list.append(ndcg3)
        mae_list.append(mae)
    return np.mean(ndcg1_list), np.mean(ndcg3_list), np.mean(mae_list)


def evaluate(epoch, args, model_dir='save_models'):
    if args.use_score:
        model = Meta_score(args).cuda()
    else:
        model = Meta(args).cuda()
    # todo: auto check epoch
    ckpt_path = os.path.join(model_dir, 'epoch_{}.pt'.format(epoch))
    model.load_state_dict(torch.load(ckpt_path))
    test_dataset = DataLoader(MetaTrainData(args, split="test"), batch_size=1, num_workers=args.num_workers)
    if args.use_score:
        ndcg1, ndcg3, mae = evaluate_test_score(args, model, test_dataset)
        print('NDCG1: {}, NDCG3: {}, MAE: {}'.format(ndcg1, ndcg3, mae))
        logging.info('NDCG1: {}, NDCG3: {}, MAE: {}'.format(ndcg1, ndcg3, mae))
    else:
        hr, ndcg1, ndcg3 = evaluate_test(args, model, test_dataset)
        print('HR: {}, NDCG1: {}, NDCG3: {}'.format(hr, ndcg1, ndcg3))
        logging.info('HR: {}, NDCG1: {}, NDCG3: {}'.format(hr, ndcg1, ndcg3))
    

if __name__ == '__main__':
    args = parse_args()
    # set random seed
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if args.dataset == 'movie':
        args.n_users = 6040 + 1
        args.n_items = 3947 + 1 + 2000
        args.data_root = 'dataset/movielens/ml-1m'
    else:
        args.n_users = 14356
        args.n_items = 15885
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    os.makedirs(args.save_dir, exist_ok=True)
    logging.getLogger().setLevel(logging.INFO)
    if args.eval == True:
        logging.basicConfig(filename=os.path.join(args.save_dir, 'evaluation.log'))
        evaluate(args.start_epoch-1, args, model_dir=args.save_dir)
    else:
        logging.basicConfig(filename=os.path.join(args.save_dir, 'train.log'))
        if args.use_score:
            model = train_score(args, save_path=args.save_dir)
        else:
            model = train(args, save_path=args.save_dir)