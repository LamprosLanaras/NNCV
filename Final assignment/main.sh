# To run a different experiment:
# - Change the training script
# - Change --variant and hyperparameters
# - Submit again with: sbatch jobscript_slurm.sh

wandb login

# -----------------------------
# Peak Performance training
# Variants: baseline, augsegformer, uperformer, auxlovasz_uperformer
# -----------------------------

python3 train_peak.py \
    --variant auxlovasz_uperformer \
    --data-dir ./data/cityscapes \
    --batch-size 2 \
    --epochs 100 \
    --lr 0.00006 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id final-peak-run

# -----------------------------
# Efficiency training example
# Variants: fastscnn, c_fastscnn, kd_c_fastscnn
# Uncomment this block and comment out the peak block above to use it.
# -----------------------------

# python3 train_efficiency.py \
#     --variant kd_c_fastscnn \
#     --data-dir ./data/cityscapes \
#     --batch-size 8 \
#     --epochs 100 \
#     --lr 0.001 \
#     --num-workers 10 \
#     --seed 42 \
#     --teacher-weights ./weights/segformer_teacher.pt \
#     --experiment-id final-efficiency-run