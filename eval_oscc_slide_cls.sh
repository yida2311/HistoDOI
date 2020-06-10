export CUDA_VISIBLE_DEVICE=0 \

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python scripts/eval_histo_slide_cls.py \
--n_class 2 \
--data_path_val "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/cls_5x/val_224/"  \
--meta_path_val "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/cls_5x/subslide_info_val_224.json"  \
--log_path "results/logs/" \
--output_path "results/predictions/" \
--task_name "resnet50-bin_cls-224-1e-4-5.25" \
--batch_size 64 \
--num_workers 2 \
--evaluation \
--ckpt_path "results/saved_models/resnet50-bin_cls-224-1e-4-5.25/resnet50-bin_cls-224-1e-4-5.25-49-0.9582271801797669.pth"
