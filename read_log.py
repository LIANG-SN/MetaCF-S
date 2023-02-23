import numpy as np
with open('results/test_movie_fewshot/evaluation.log', 'r') as f:
    lines = f.read().splitlines()

count = 0
ndcg3=ndcg1=mae=0

# for row in range(2663-1, 2945-1, 2):
for row in range(1-1, 3891-1, 2):
    tmp = lines[row].split(',')[1] # 1:ndcg1, 2:ndcg3, 3:mae
    ndcg1 += np.float32(tmp[tmp.index(':')+2:])
    tmp = lines[row].split(',')[2] # 1:ndcg1, 2:ndcg3, 3:mae
    ndcg3 += np.float32(tmp[tmp.index(':')+2:])
    tmp = lines[row].split(',')[3] # 1:ndcg1, 2:ndcg3, 3:mae
    mae += np.float32(tmp[tmp.index(':')+2:])
    count += 1
print(ndcg1/(count), ndcg3/(count), mae/count)
