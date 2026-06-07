**Project: Digital Twin Fault Diagnosis — Transfer Learning Optimized**

This repository contains code and resources for transfer-learning based fault diagnosis using a digital-twin (DTR) Simulink model and PyTorch training scripts.

**Contents**
- **Model:** The Simulink model is in [dtr digital model simulink/trainSimTestReal.mlx](dtr%20digital%20model%20simulink/trainSimTestReal.mlx#L1).
- **Training scripts:** See [scripts/train_transfer_learning_cv5.py](scripts/train_transfer_learning_cv5.py#L1) and [scripts/train_ai_pytorch_main.py](scripts/train_ai_pytorch_main.py#L1).
- **Submission script (La Ruche):** [scripts/submit_cv5.sh](scripts/submit_cv5.sh#L1).
- **Dataset:** Train/test MAT files are in [mydataset/my_dataset_train.mat](mydataset/my_dataset_train.mat#L1) and [mydataset/my_dataset_test.mat](mydataset/my_dataset_test.mat#L1).
- **Dependencies:** See [requirements.txt](requirements.txt#L1).

**Important notes**
- **Dataset size used for the Simulink DTR model:** The model in [dtr digital model simulink](dtr%20digital%20model%20simulink/trainSimTestReal.mlx#L1) was trained on a dataset containing 800 points per class.

**Quickstart — Run locally**
- **Install dependencies:**

```
pip install -r requirements.txt
```

- **Run training (example):**

```
python scripts/train_transfer_learning_cv5.py
```

**Run on La Ruche (using the provided submission script)**
- The repository includes a submission script to run the cross-validation job on La Ruche: [scripts/submit_cv5.sh](scripts/submit_cv5.sh#L1).
- From the repository root on La Ruche, submit the job with:

```
bash scripts/submit_cv5.sh
```

- The script is intended to handle job submission (adjust the script if your cluster requires specific `sbatch` or environment settings).

**Saved models**
- Example trained checkpoints are in `scripts/` (e.g. `best_finetuned_cv.pt`, `best_pretrained_cv.pt`).

