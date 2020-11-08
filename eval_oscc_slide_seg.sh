export CUDA_VISIBLE_DEVICE=0 \

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python eval_slide_unet.py \
--n_class 4 \
--img_path_val "/media/ldy/7E1CA94545711AE6/OSCC_test/5x_1600/train_1600/"  \
--meta_path_val "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile-v2/5x_1600/tile_info_train_1600.json"  \
--slide_mask_path "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile-v2/slide/std_mask_rgb/" \
--log_path "results-v2/logs/" \
--output_path "results-v2/predictions/" \
--npy_path "results-v2/slide_npy/" \
--task_name "unet-dan-ce-cos120-8.16-test-slide" \
--batch_size 2 \
--num_workers 2 \
--evaluation \
--ckpt_path "results-v2/saved_models/unet-dan-ce-cos120-8.16/unet-dan-ce-cos120-8.16-113-0.87969.pth" \


