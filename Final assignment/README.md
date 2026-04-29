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
### 🔑 Environment Variables
This project uses Weights & Biases for logging. Before running training, you must set up your environment variables:

1. Copy the template file: `cp .env.example .env`
2. Open the new `.env` file and replace `your_wandb_api_key_here` with your actual API key.

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

## 🐳 Building a Docker Image
Before running local inference or submitting into the challenge server, build the Docker image from the root of the repository:

```bash
docker build -t nncv-inference .
```

## 🚀 Peak Performance

The Peak Performance track focuses on improving segmentation accuracy.

We start with the original Segformer-B5 and enhance it with various augmentantions (AugSegformer). Then, we couple the encoder of Segformer-B5 with an UPerNet decoder (UPerformer). In this new architecture, we attach an additional intermediate loss head and tweak the loss function (Aux-Lovász UPerFormer).

Model progression:

```text
SegFormer-B5 (baseline) → AugSegformer → UPerFormer → Aux-Lovász UPerFormer
```

### Supported Variants (Code)

```text
baseline
augsegformer
uperformer
auxlovasz_uperformer
```


### Example Training Command

This command is intended to be executed inside the HPC cluster through `main.sh`:

```bash
python train_peak.py --variant auxlovasz_uperformer --experiment-id final-peak-run
```


### Example Local Inference Command

#### With Docker
To run inference locally, we mount the `local_data` and `local_output` directories into the container so the model can read the inputs and save the predictions back to your host machine.







```bash
docker run --rm --gpus all \
    -v "$(pwd)/local_data:/app/local_data" \
    -v "$(pwd)/local_output:/app/local_output" \
    nncv-inference predict_peak.py \
    --variant auxlovasz_uperformer \
    --weights_path ./weights/best_peak.pt \
    --input_dir ./local_data \
    --output_dir ./local_output
```

#### Without Docker

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

We use a Fast-SCNN as baseline and then stick with a compressed Fast-SCNN variant architecture (C-FastSCNN). To this compressed variant, we perform Knowledge Distillation from a SegFormer-B5 teacher (KD-C-FastSCNN).

Model progression:

```text
Fast-SCNN (baseline) → C-FastSCNN → KD-C-FastSCNN
```



### Supported Variants (Code)

```text
fastscnn
c_fastscnn
kd_c_fastscnn
```

### Example Training Command

This command is intended to be executed inside the HPC cluster through `main.sh`:

```bash
python train_efficiency.py --variant kd_c_fastscnn --teacher-weights ./weights/segformer_teacher.pt
```

### Example Local Inference Command

#### With Docker

```bash
docker run --rm --gpus all \
    -v "$(pwd)/local_data:/app/local_data" \
    -v "$(pwd)/local_output:/app/local_output" \
    nncv-inference predict_efficiency.py \
    --variant kd_c_fastscnn \
    --weights_path ./weights/best_efficiency.pt \
    --input_dir ./local_data \
    --output_dir ./local_output
```

#### Without Docker

```bash
python predict_efficiency.py \
    --variant kd_c_fastscnn \
    --weights_path ./weights/best_efficiency.pt \
    --input_dir ./local_data \
    --output_dir ./local_output
```



## ✅ Main Entry Points

```text
train_peak.py            # Peak-performance training
train_efficiency.py      # Efficiency/KD training
predict_peak.py          # Peak-performance inference
predict_efficiency.py    # Efficiency inference
main.sh                  # HPC command router
jobscript_slurm.sh       # Slurm submission script
Dockerfile               # Submission image
```

## Author and Verification

- TU/e email: l.lanaras@student.tue.nl


### Peak Performance Models

| Model | Challenge Server Name |
|---|---|
| SegFormer baseline | `LaNet_PP_v1` |
| AugSegFormer | `LaNet_PP_v3` |
| UPerFormer | `LaNet_Lovasz_Uper` |
| Aux-Lovász UPerFormer | `AuxLovászUperFormer` |

### Efficiency Models

| Model | Challenge Server Name |
|---|---|
| Fast-SCNN baseline | `LaNet_RT_v2.6` |
| C-FastSCNN | `LaNet_t4_no_kd` |
| KD-C-FastSCNN | `LaNet_t4_kd_0.2_0.9` |

## Top-1 submission

Our **KD-C-FastSCNN** model ranked the highest in the 'Efficiency' benchmark:

```text
LaNet_t4_kd_0.2_0.9 🏆