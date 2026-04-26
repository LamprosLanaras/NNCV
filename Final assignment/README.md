# On Neural Networks for Semantic Segmentation

This repository contains the implementation pipeline for the NNCV final assignment.
The work is divided into two research tracks:

- **Peak Performance**
- **Efficiency** 

Training is intended to run on an **HPC cluster** using **Slurm**. Final evaluation is performed through **Docker challenge submissions**.

---

## 📂 Project Structure

```text
NNCV_Final_Assignment/
├── models/
│   ├── peak_performance/      # SegFormer, UPerNet, Aux-Lovász UPerFormer
│   └── efficiency/            # Fast-SCNN and compressed Fast-SCNN variants
├── weights/                   # Trained .pt checkpoints
├── local_data/                # Small local image folder for inference testing
├── local_output/              # Local prediction outputs
├── train_peak.py              # Trainer for peak-performance models
├── train_efficiency.py        # Trainer for efficiency and KD models
├── predict_peak.py            # Inference for peak-performance models
├── predict_efficiency.py      # Inference for efficiency models
├── jobscript_slurm.sh         # Slurm job script
├── main.sh                    # HPC/Apptainer entry point
├── Dockerfile                 # Docker submission image definition
└── requirements.txt           # Python dependencies
```

---

## 📚 Documentation

Detailed setup and submission instructions are kept in separate guide files:

```text
SLURM_README.md              # Running jobs on the Slurm HPC cluster
CHALLENGE_SUBMISSION.md      # Building, testing, exporting, and submitting Docker images
```


## 📁 Data Structure

The training scripts expect the Cityscapes dataset to follow this structure:

```text
data/cityscapes/
├── leftImg8bit/
│   ├── train/
│   └── val/
└── gtFine/
    ├── train/
    └── val/
```

The dataset is downloaded and prepared on the HPC following the Slurm setup guide.

---

## 🚀 Peak Performance

The Peak Performance track focuses on improving segmentation accuracy.

Model progression:

```text
SegFormer-B5 → AugSegformer → UPerFormer → Aux-Lovász UPerFormer
```

### Supported Variants

```text
baseline
augsegformer
uperformer
auxlovasz_uperformer
```


### Example Training Command

This command is intended to be executed inside the HPC container through `main.sh`:

```bash
python train_peak.py --variant auxlovasz_uperformer --experiment-id final-peak-run
```

### Example Local Inference Command

```bash
python predict_peak.py \
    --variant auxlovasz_uperformer \
    --weights_path ./weights/best_peak.pt \
    --input_dir ./local_data \
    --output_dir ./local_output
```

---

## ⚡ Efficiency

The Efficiency track focuses on real-time semantic segmentation.

We use a Fast-SCNN as baseline and the stick with a compressed Fast-SCNN variant architecture. To this compressed variant, we perform Knowledge Distillation from a SegFormer-B5 teacher.

### Supported Variants

```text
fastscnn
c_fastscnn
kd_c_fastscnn
```

### Example Training Command

This command is intended to be executed inside the HPC container through `main.sh`:

```bash
python train_efficiency.py --variant kd_c_fastscnn --teacher-weights ./weights/segformer_teacher.pt
```

### Example Local Inference Command

```bash
python predict_efficiency.py \
    --variant kd_c_fastscnn \
    --weights_path ./weights/best_efficiency.pt \
    --input_dir ./local_data \
    --output_dir ./local_output
```

---



## ✅ Main Entry Points

```text
train_peak.py            # Peak-performance training
train_efficiency.py      # Efficiency/KD training
predict_peak.py          # Peak-performance inference
predict_efficiency.py    # Efficiency inference
main.sh                  # HPC command router
jobscript_slurm.sh       # Slurm submission script
Dockerfile               # Challenge submission image
```