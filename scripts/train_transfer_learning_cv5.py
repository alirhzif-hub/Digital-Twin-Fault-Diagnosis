#!/usr/bin/env python
# coding: utf-8
"""
Transfer Learning — 5-Fold Stratified Cross-Validation
Phase 1: Pre-train on simulated data (une seule fois)
Phase 2: Fine-tune sur chaque fold des données réelles
"""

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import scipy.io
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Config ───────────────────────────────────────────────────────────────────
categories = [
    'Healthy',
    'Motor_1_Stuck', 'Motor_1_Steady_state_error',
    'Motor_2_Stuck', 'Motor_2_Steady_state_error',
    'Motor_3_Stuck', 'Motor_3_Steady_state_error',
    'Motor_4_Stuck', 'Motor_4_Steady_state_error',
]

N_FOLDS    = 5
SEED       = 42

P1_EPOCHS  = 250
P1_LR      = 1e-3
P1_BATCH   = 32
P1_PATIENCE= 30

P2A_EPOCHS = 50
P2A_LR     = 1e-3
P2A_BATCH  = 16

P2B_EPOCHS = 100
P2B_LR     = 1e-4
P2B_BATCH  = 16

MODEL_SAVE_PATH   = "best_pretrained_cv.pt"
FINETUNE_SAVE_PATH= "best_finetuned_cv.pt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ─── Seed ─────────────────────────────────────────────────────────────────────
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ─── Data loading ─────────────────────────────────────────────────────────────
def load_mat_to_tensor(mat, x_key, y_key, mean=None, std=None):
    data_X = mat[x_key][0]
    data_Y = mat[y_key][0]
    X = torch.tensor(
        np.array([data_X[i] for i in range(len(data_X))]),
        dtype=torch.float32
    )
    residual = X[:, :, :3] - X[:, :, 3:6]
    X[:, :, 3:6] = residual
    if mean is None:
        mean = X.mean(dim=(0, 1), keepdim=True)
        std  = X.std(dim=(0, 1), keepdim=True)
    std[std == 0] = 1e-8
    X = (X - mean) / std
    cat2idx = {c: i for i, c in enumerate(categories)}
    Y_raw = np.array([data_Y[i] for i in range(len(data_Y))]).flatten()
    Y = torch.tensor([cat2idx[c] for c in Y_raw], dtype=torch.int64)
    return X.to(device), Y.to(device), mean, std

mat_train = scipy.io.loadmat('../mydataset/my_dataset_train.mat')
mat_test  = scipy.io.loadmat('../mydataset/my_dataset_test.mat')

X_sim, Y_sim, sim_mean, sim_std = load_mat_to_tensor(mat_train, 'X_array', 'y_array')
X_real, Y_real, _, _ = load_mat_to_tensor(mat_test, 'X_test_array', 'y_test_array',
                                           mean=sim_mean, std=sim_std)

input_size  = X_sim.shape[2]
seq_len     = X_sim.shape[1]
num_classes = len(categories)

print(f"Simulated: {X_sim.shape}  |  Real: {X_real.shape}")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def criterion(logits, labels):
    return F.nll_loss(F.log_softmax(logits, dim=1), labels)

def run_epoch(model, X, Y, batch_size, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses, preds_all, labels_all = [], [], []
    with torch.set_grad_enabled(is_train):
        for i in range(0, len(X), batch_size):
            xb = X[i:i+batch_size].view(-1, seq_len, input_size)
            yb = Y[i:i+batch_size]
            out = model(xb)
            loss = criterion(out, yb)
            if is_train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
            preds_all.append(torch.argmax(out, 1).cpu())
            labels_all.append(yb.cpu())
    preds_all  = torch.cat(preds_all)
    labels_all = torch.cat(labels_all)
    return np.mean(losses), accuracy_score(labels_all, preds_all)

# ─── Phase 1: Pre-train sur données simulées (une seule fois) ─────────────────
from models import ComplexCNN
from sklearn.model_selection import StratifiedShuffleSplit

print("\n" + "="*60)
print("PHASE 1 — Pre-training sur données simulées")
print("="*60)

model = ComplexCNN(input_size=input_size, num_classes=num_classes, num_channels=128).to(device)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
idx_train, idx_val = next(sss.split(X_sim.cpu(), Y_sim.cpu()))
X_sim_train, X_sim_val = X_sim[idx_train], X_sim[idx_val]
Y_sim_train, Y_sim_val = Y_sim[idx_train], Y_sim[idx_val]

optimizer_p1 = optim.Adam(model.parameters(), lr=P1_LR)
scheduler_p1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer_p1, mode='min', factor=0.5, patience=20)
best_val_loss = float('inf')
no_improve    = 0

for epoch in range(P1_EPOCHS):
    tr_loss, tr_acc = run_epoch(model, X_sim_train, Y_sim_train, P1_BATCH, optimizer_p1)
    vl_loss, vl_acc = run_epoch(model, X_sim_val,   Y_sim_val,   P1_BATCH)
    scheduler_p1.step(vl_loss)
    if vl_loss < best_val_loss:
        best_val_loss = vl_loss; no_improve = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    else:
        no_improve += 1
    if no_improve >= P1_PATIENCE:
        print(f"Early stopping at epoch {epoch+1}")
        break
    if (epoch+1) % 50 == 0 or epoch == 0:
        print(f"Ep {epoch+1:3d} | tr_loss={tr_loss:.4f} vl_loss={vl_loss:.4f} | "
              f"tr_acc={tr_acc*100:.1f}% vl_acc={vl_acc*100:.1f}%")

print(f"Best val loss: {best_val_loss:.4f}")

# ─── 5-Fold Cross-Validation sur données réelles ──────────────────────────────
print("\n" + "="*60)
print(f"5-FOLD STRATIFIED CROSS-VALIDATION sur données réelles")
print("="*60)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
X_real_np = X_real.cpu().numpy()
Y_real_np = Y_real.cpu().numpy()

all_acc, all_f1 = [], []
cm_sum = np.zeros((num_classes, num_classes), dtype=int)

for fold, (idx_ft, idx_test) in enumerate(skf.split(X_real_np, Y_real_np)):
    print(f"\n{'─'*50}")
    print(f"FOLD {fold+1}/{N_FOLDS}  — train={len(idx_ft)} test={len(idx_test)}")
    print(f"{'─'*50}")

    X_real_ft,   Y_real_ft   = X_real[idx_ft],   Y_real[idx_ft]
    X_real_test, Y_real_test = X_real[idx_test], Y_real[idx_test]

    # Charger les poids pré-entraînés
    model = ComplexCNN(input_size=input_size, num_classes=num_classes, num_channels=128).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))

    # Phase 2a — backbone gelé
    for param in model.tcn.parameters():
        param.requires_grad = False
    optimizer_p2a = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=P2A_LR)
    best_ft_loss = float('inf')

    for epoch in range(P2A_EPOCHS):
        ft_loss, ft_acc = run_epoch(model, X_real_ft, Y_real_ft, P2A_BATCH, optimizer_p2a)
        if ft_loss < best_ft_loss:
            best_ft_loss = ft_loss
            torch.save(model.state_dict(), FINETUNE_SAVE_PATH)
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  2a Ep {epoch+1:3d}/{P2A_EPOCHS} | ft_loss={ft_loss:.4f} ft_acc={ft_acc*100:.1f}%")

    # Phase 2b — fine-tuning complet
    for param in model.parameters():
        param.requires_grad = True
    model.load_state_dict(torch.load(FINETUNE_SAVE_PATH, weights_only=True))
    optimizer_p2b = optim.Adam(model.parameters(), lr=P2B_LR)
    scheduler_p2b = optim.lr_scheduler.CosineAnnealingLR(optimizer_p2b, T_max=P2B_EPOCHS)

    for epoch in range(P2B_EPOCHS):
        ft_loss, ft_acc = run_epoch(model, X_real_ft, Y_real_ft, P2B_BATCH, optimizer_p2b)
        scheduler_p2b.step()
        if ft_loss < best_ft_loss:
            best_ft_loss = ft_loss
            torch.save(model.state_dict(), FINETUNE_SAVE_PATH)
        if (epoch+1) % 20 == 0 or epoch == 0:
            lr = optimizer_p2b.param_groups[0]['lr']
            print(f"  2b Ep {epoch+1:3d}/{P2B_EPOCHS} | lr={lr:.2e} ft_loss={ft_loss:.4f} ft_acc={ft_acc*100:.1f}%")

    # Evaluation sur le fold de test
    model.load_state_dict(torch.load(FINETUNE_SAVE_PATH, weights_only=True))
    model.eval()
    with torch.no_grad():
        out   = model(X_real_test.view(-1, seq_len, input_size))
        preds = torch.argmax(out, 1).cpu()

    acc = accuracy_score(Y_real_test.cpu(), preds)
    f1  = f1_score(Y_real_test.cpu(), preds, average='weighted')
    cm  = confusion_matrix(Y_real_test.cpu(), preds, labels=list(range(num_classes)))

    all_acc.append(acc)
    all_f1.append(f1)
    cm_sum += cm

    print(f"\n  Fold {fold+1} — Accuracy={acc*100:.2f}%  F1={f1:.4f}")

# ─── Résumé final ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"RÉSUMÉ 5-FOLD CROSS-VALIDATION")
print(f"{'='*60}")
for i, (acc, f1) in enumerate(zip(all_acc, all_f1)):
    print(f"  Fold {i+1} : Accuracy={acc*100:.2f}%  F1={f1:.4f}")
print(f"{'─'*60}")
print(f"  Moyenne : Accuracy={np.mean(all_acc)*100:.2f}% ± {np.std(all_acc)*100:.2f}%")
print(f"  Moyenne : F1={np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
print(f"  Min     : {np.min(all_acc)*100:.2f}%")
print(f"  Max     : {np.max(all_acc)*100:.2f}%")

# Matrice de confusion cumulée
plt.figure(figsize=(12, 9))
sns.heatmap(cm_sum, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories)
plt.title(f'Confusion Matrix — 5-Fold Cross-Validation\n'
          f'Mean Acc={np.mean(all_acc)*100:.2f}% ± {np.std(all_acc)*100:.2f}%')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.tight_layout()
plt.savefig("cm_cv5fold.png", dpi=100)
print("\nMatrice de confusion sauvegardée : cm_cv5fold.png")
