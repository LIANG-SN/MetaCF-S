python main.py \
    --dataset movie \
    --save_dir results/test_movie_cold_item \
    --cuda 5 \
    --test_k_shot 5 \
    --eval \
    --num_workers 0 \
    --use_score \
    --start_epoch 1 \
    # --original_model 
    # --dot_prod


# python main.py \
#     --dataset movie \
#     --save_dir results/test_movie_rank3 \
#     --cuda 5 \
#     --test_k_shot 5 \
#     --eval \
#     --topk 3 \
#     --num_workers 8 \