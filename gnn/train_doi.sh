python main.py  \
--n_class 3 \
--max_num_nodes [10, 10, 30] \
--min_node_size [20] \
--num_edges_per_class 5 \
--node_resize 32 \
--scheduler 'poly' \
--img_path_train 'results-v2/slide_npy/unet-dan-ce-cos120-8.16-train-slide/' \
--mask_path_train '/media/ldy/7E1CA94545711AE6/OSCC/mask_1.25x_v2/std_mask/' \
--img_path_val 'results-v2/slide_npy/unet-dan-ce-cos120-8.16-val-slide/' \
--mask_path_val '/media/ldy/7E1CA94545711AE6/OSCC/mask_1.25x_v2/std_mask/' \
--model_path 'results-gcn/saved_models/' \
--output_path 'results-gcn/predictions/' \
--log_path 'results-gcn/logs/' \
--task_name 'gcn-1.25x-9.6' \
--evaluation \
--batch_size 4 \
--num_workers 4 \
--epochs 200 \
--lr 1e-3 \
--alpha 1.0 \


