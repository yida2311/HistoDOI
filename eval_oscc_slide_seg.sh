export CUDA_VISIBLE_DEVICE=0 \

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python scripts/eval_histo_slide_seg.py \
--n_class 4 \
--img_path_val "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/5x_1600/train_1600/"  \
--mask_path_val "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/5x_1600/train_mask_1600/"  \
--meta_path_val "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/5x_1600/tile_info_train_1600.json"  \
--log_path "results/logs/seg/" \
--output_path "results/predictions/seg/" \
--task_name "unet-resnet34-seg-1024-1e-4-poly-6.20-train-slide" \
--batch_size 2 \
--num_workers 2 \
--evaluation \
--ckpt_path "results/saved_models/seg/unet-resnet34-seg-1024-1e-4-poly-6.20/unet-resnet34-seg-1024-1e-4-poly-6.20-115-0.8402176034008562.pth"


