export CUDA_VISIBLE_DEVICE=0 \

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python scripts/train_histo_cls.py \
--n_class 3 \
--warmup_epochs 1 \
--scheduler 'poly' \
--data_path_train "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/5x_224/train_224/" \
--meta_path_train "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/5x_224/train_224.csv"  \
--data_path_val "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/5x_224/val_224/"  \
--meta_path_val "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC-Tile/5x_224/val_224.csv"  \
--model_path "results/saved_models/cls/" \
--log_path "results/logs/cls/" \
--task_name "xxx" \
--batch_size 8 \
--num_workers 2 \
--epochs 100 \
--lr 1e-2 \
