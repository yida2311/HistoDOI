export CUDA_VISIBLE_DEVICE=0 \

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python scripts/eval_histo_slide_seg.py \
--n_class 4 \
--img_path_val "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/seg_5x/val_bd_224/"  \
--mask_path_val "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/seg_5x/val_bd_mask_224/"  \
--meta_path_val "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/seg_5x/subslide_info_val_224.json"  \
--log_path "results/logs/" \
--output_path "results/predictions/" \
--task_name "unet-seg-bd-224-2e-4-6.3" \
--batch_size 64 \
--num_workers 2 \
--evaluation \
--ckpt_path "results/saved_models/seg/unet-seg-bd-224-2e-4-6.3/unet-seg-bd-224-2e-4-6.3-79-0.8046013115498818.pth"
