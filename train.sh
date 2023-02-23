python main.py \
    --dataset movie \
    --save_dir results/test_movie_cold_item \
    --cuda 5 \
    --test_k_shot 5 \
    --use_score \
    --num_workers 0 \
    --lr_inner_init 4e-4 \
    --lr_meta 1e-4 \
    --start_epoch 0 \
    # --original_model 
    # --dot_prod