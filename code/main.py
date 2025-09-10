import os
import pickle
import random
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix

from torch.optim import lr_scheduler

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

# =========================
# 0. Set Random Seed
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =========================
# 1. Configuration
# =========================
TRAIN_PKL = "../datasets/github10_20250303.pkl"
TEST_PKL  = "../datasets/github10_20250502.pkl"

# --- Log and Model Output ---
SAVE_DIR = "./output"
os.makedirs(SAVE_DIR, exist_ok=True)

# File Path Configuration
LABEL_ENCODER_PATH = os.path.join(SAVE_DIR, "label_encoder.pkl")
MODEL_PATH         = os.path.join(SAVE_DIR, "flow_transformer.pt")
PROTOTYPES_PATH    = os.path.join(SAVE_DIR, "prototypes.pt")
CONFUSION_WEIGHTS_PATH = os.path.join(SAVE_DIR, "confusion_weights.pt")
THRESHOLDS_PATH    = os.path.join(SAVE_DIR, "dynamic_thresholds.pt") # Save dynamic thresholds

# Training Hyperparameters
LR          = 1e-4
EPOCHS      = 500
BATCH_SIZE  = 64

# Transformer Model Hyperparameters
D_MODEL = 256
N_HEAD = 8
N_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# --- Dynamic Threshold Hyperparameters ---
THRESHOLD_PERCENTILE = 99   # Percentile for calculating the base threshold
TOP_K_NEIGHBORS = 5         # Number of nearest neighbors to consider for inter-class distance
THRESHOLD_SCALE = 0.1       # Maximum scaling factor for dynamic threshold adjustment (e.g., 0.1 means the threshold can be scaled down by at most 10%)


# t-SNE Visualization Hyperparameters
TSNE_MAX_SAMPLES = 1000
TSNE_PERPLEXITY = 40

# =========================
# 2. Data Loading and Preprocessing
# =========================
def log_normalize(seq):
    arr = np.array(seq, dtype=np.float32)
    return torch.tensor(np.log1p(arr), dtype=torch.float32)

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_seqs, lengths, labels

class FlowDataset(Dataset):
    def __init__(self, X: List[torch.Tensor], y: List[str], le: LabelEncoder, known_class: List[str]):
        self.X = X
        self.le = le
        self.known_class = set(known_class)
        y_mapped = [le.transform([l])[0] if l in self.known_class else -1 for l in y]
        self.y = torch.tensor(y_mapped, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if x.dim() == 1:
            x = torch.stack([x, torch.zeros_like(x)], dim=-1)
        return x, self.y[idx]

# =========================
# 3. Transformer Model and Adaptive Loss
# =========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class FlowTransformer(nn.Module):
    def __init__(self, input_dim, d_model, n_head, n_layers, dim_feedforward, dropout):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, lengths):
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        seq_mask = torch.arange(seq_len, device=x.device)[None, :] >= lengths[:, None]
        src_key_padding_mask = torch.cat([cls_mask, seq_mask], dim=1)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        cls_output = output[:, 0, :]
        final_feature = self.layer_norm(cls_output)
        return final_feature

class AdaptivePrototypicalLoss(nn.Module):
    def __init__(self, num_classes: int):
        super(AdaptivePrototypicalLoss, self).__init__()
        self.num_classes = num_classes
        self.register_buffer('class_weights', torch.ones(num_classes))

    def forward(self, features, labels):
        classes = torch.unique(labels)
        # Handle case where a batch might not have enough classes to form prototypes
        if len(features) == 0 or len(classes) == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
            
        prototypes = torch.stack([features[labels == c].mean(dim=0) for c in classes])
        dists = torch.cdist(features, prototypes)
        logits = -dists
        
        map_labels = torch.zeros_like(labels)
        for i, c in enumerate(classes):
            map_labels[labels == c] = i
            
        batch_weights = self.class_weights[classes]
        criterion = nn.CrossEntropyLoss(weight=batch_weights)
        loss = criterion(logits, map_labels)
        return loss

    @torch.no_grad()
    def update_weights(self, all_features: torch.Tensor, all_labels: torch.Tensor):
        print("\nUpdating adaptive loss weights based on class confusion...")
        device = all_features.device
        
        # Ensure there are features to process
        if len(all_features) == 0:
            print("No features to update weights.")
            return

        prototypes = torch.stack([all_features[all_labels == c].mean(dim=0) for c in range(self.num_classes)])
        confusion_scores = torch.zeros(self.num_classes, device=device)
        proto_dists = torch.cdist(prototypes, prototypes)
        
        for c in range(self.num_classes):
            features_c = all_features[all_labels == c]
            if len(features_c) == 0: continue
            
            intra_dist = torch.cdist(features_c, prototypes[c].unsqueeze(0)).mean()
            inter_dists_c = proto_dists[c].clone()
            inter_dists_c[c] = float('inf')
            
            if torch.isinf(inter_dists_c).all(): # Handle case with only one class
                min_inter_dist = 1.0
            else:
                min_inter_dist = inter_dists_c.min()

            confusion_scores[c] = intra_dist / (min_inter_dist + 1e-8)
            
        new_weights = F.softmax(confusion_scores, dim=0) * self.num_classes
        self.class_weights = 0.5 * self.class_weights + 0.5 * new_weights
        # print("Updated Class Weights (Confusion-based):")
        # weights_np = self.class_weights.cpu().numpy()
        # for i, w in enumerate(weights_np):
        #     print(f"  Class {i}: {w:.4f}")

# =========================
# 4. Helper Functions
# =========================
def extract_features_global(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracts features for the entire dataset."""
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for X_batch, lengths, y_batch in dataloader:
            features = model(X_batch.to(device), lengths.to(device))
            all_features.append(features.cpu())
            all_labels.append(y_batch)
    return torch.cat(all_features), torch.cat(all_labels)


# =========================
# 5. Training and Evaluation Function
# =========================
def train_eval_model(train_loader, test_loader, num_classes, le: LabelEncoder, device: torch.device):
    model = FlowTransformer(
        input_dim=2, d_model=D_MODEL, n_head=N_HEAD, n_layers=N_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
    ).to(device)
    
    criterion = AdaptivePrototypicalLoss(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # -------- Training --------
    for epoch in range(EPOCHS):
        model.train()
        total_loss, n_samples = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for X_batch, lengths, y_batch in pbar:
            X_batch, lengths, y_batch = X_batch.to(device), lengths.to(device), y_batch.to(device)
            mask = y_batch >= 0
            if mask.sum() < 2 or len(torch.unique(y_batch[mask])) < 2: continue # Need at least 2 samples from 2 classes
            
            optimizer.zero_grad()
            features = model(X_batch[mask], lengths[mask])
            loss = criterion(features, y_batch[mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * mask.sum().item()
            n_samples += mask.sum().item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Epoch {epoch+1}/{EPOCHS}] Avg Loss={total_loss/max(1,n_samples):.6f} | lr={current_lr:.6f}")
        
        if (epoch + 1) % 1 == 0:
            train_features_for_weights, train_labels_for_weights = extract_features_global(model, train_loader, device)
            known_mask_weights = train_labels_for_weights != -1
            criterion.update_weights(train_features_for_weights[known_mask_weights].to(device), train_labels_for_weights[known_mask_weights].to(device))

    torch.save(model.state_dict(), MODEL_PATH)
    final_weights = criterion.class_weights.cpu()
    torch.save(final_weights, CONFUSION_WEIGHTS_PATH)
    print(f"Final confusion-based weights saved to {CONFUSION_WEIGHTS_PATH}")

    # --- Calculate Resources for Evaluation (Core: Dynamic Thresholds) ---
    print("\nCalculating resources for evaluation with DYNAMIC thresholds...")
    train_features, train_labels = extract_features_global(model, train_loader, device)
    known_mask = train_labels != -1
    train_features, train_labels = train_features[known_mask], train_labels[known_mask]
    
    # 1. Calculate Global Prototypes
    global_prototypes = torch.stack([train_features[train_labels == c].mean(dim=0) for c in range(num_classes)])
    torch.save(global_prototypes, PROTOTYPES_PATH)
    print(f"Global prototypes saved to {PROTOTYPES_PATH}")

    # 2. Calculate a 'difficulty score' S_c for each class
    print(f"\nCalculating class-specific difficulty scores (Top-{TOP_K_NEIGHBORS} neighbors)...")
    intra_dists = torch.zeros(num_classes)
    inter_dists = torch.zeros(num_classes)
    
    proto_dists_matrix = torch.cdist(global_prototypes, global_prototypes) # Calculate distances between all prototypes

    for c in range(num_classes):
        # a. Calculate intra-class spread (Intra_c): Average distance from samples of class c to their prototype.
        features_c = train_features[train_labels == c]
        if len(features_c) > 0:
            intra_dists[c] = torch.cdist(features_c, global_prototypes[c].unsqueeze(0)).mean()
        
        # b. Calculate inter-class proximity (Inter_c): Average distance from class c's prototype to the Top-K nearest prototypes of other classes.
        # Sort and take the 1st to K+1th elements (the 0th is the prototype itself, with distance 0)
        if num_classes > TOP_K_NEIGHBORS:
            k_nearest_dists = torch.topk(proto_dists_matrix[c], k=TOP_K_NEIGHBORS + 1, largest=False).values[1:]
        else: # Handle case with fewer classes than K
            k_nearest_dists = torch.topk(proto_dists_matrix[c], k=num_classes, largest=False).values[1:]

        inter_dists[c] = k_nearest_dists.mean() if len(k_nearest_dists) > 0 else 1e9

    # c. Final score S_c = Intra_c / Inter_c. A higher score means the class is more spread out internally and closer to its neighbors, making it harder to distinguish.
    difficulty_scores = intra_dists / (inter_dists + 1e-8)
    print("Difficulty Scores (Intra/Inter):")
    for i, s in enumerate(difficulty_scores):
        print(f"  Class {i} ({le.inverse_transform([i])[0]}): {s:.4f}")

    # 3. Calculate Base Thresholds (using percentiles)
    base_thresholds = torch.zeros(num_classes)
    for c in range(num_classes):
        features_c = train_features[train_labels == c]
        if len(features_c) > 0:
            dists_c = torch.cdist(features_c, global_prototypes[c].unsqueeze(0)).squeeze()
            base_thresholds[c] = torch.quantile(dists_c, THRESHOLD_PERCENTILE / 100.0)
        else:
            base_thresholds[c] = float('inf')

    # 4. Calculate Dynamic Adjustment Factors (gamma_c) based on difficulty scores
    # Goal: The higher the score S_c (more difficult), the smaller the adjustment factor, which tightens (lowers) the threshold to be more strict.
    # Normalize scores using z-score, then use the tanh function to smoothly map them to adjustment factors.
    if difficulty_scores.std() > 0:
        normalized_scores = (difficulty_scores - difficulty_scores.mean()) / difficulty_scores.std()
    else:
        normalized_scores = torch.zeros_like(difficulty_scores)
    
    adjustment_factors = 1.0 - THRESHOLD_SCALE * torch.tanh(normalized_scores)

    # 5. Calculate and Save Final Dynamic Thresholds
    final_thresholds = base_thresholds * adjustment_factors
    torch.save(final_thresholds, THRESHOLDS_PATH)

    print("\nFinal Personalized & Dynamic Thresholds:")
    for i, t in enumerate(final_thresholds):
        print(f"  Class {i} ({le.inverse_transform([i])[0]}): {t:.4f} (Base: {base_thresholds[i]:.4f}, Adj.Factor: {adjustment_factors[i]:.4f})")
    
    # -------- Closed-set and Open-set Evaluation --------
    model.eval()
    y_true_all, all_features_test = [], []
    with torch.no_grad():
        for X_batch, lengths, y_batch in tqdm(test_loader, desc="Evaluation"):
            features = model(X_batch.to(device), lengths.to(device))
            y_true_all.extend(y_batch.numpy())
            all_features_test.append(features.cpu())
            
    y_true_all = np.array(y_true_all)
    all_features_test = torch.cat(all_features_test)

    # Load resources required for evaluation
    global_prototypes_eval = torch.load(PROTOTYPES_PATH)
    thresholds_eval = torch.load(THRESHOLDS_PATH)
    
    # -- Closed-set Evaluation --
    dists_closed = torch.cdist(all_features_test, global_prototypes_eval)
    y_pred_closed_all = torch.argmin(dists_closed, dim=1).numpy()
    
    known_mask_eval = y_true_all >= 0
    y_true_closed, y_pred_closed = y_true_all[known_mask_eval], y_pred_closed_all[known_mask_eval]
    if len(y_true_closed) > 0:
        acc = accuracy_score(y_true_closed, y_pred_closed)
        pre = precision_score(y_true_closed, y_pred_closed, average="macro", zero_division=0)
        rec = recall_score(y_true_closed, y_pred_closed, average="macro", zero_division=0)
        f1  = f1_score(y_true_closed, y_pred_closed, average="macro", zero_division=0)
        print(f"\nClosed-set Results -> Acc: {acc:.4f}, Prec: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # --- Open-set Classification using Dynamic Thresholds ---
    print("\nPerforming open-set classification with DYNAMIC thresholds...")
    # 1. Find the nearest class and the corresponding minimum distance for each sample
    min_dists, preds_potential = torch.min(dists_closed, dim=1)
    
    # 2. Get the dynamic threshold for each potentially predicted class
    class_specific_thresholds = thresholds_eval[preds_potential]
    
    # 3. Check if the minimum distance exceeds the class-specific dynamic threshold
    is_unknown = min_dists > class_specific_thresholds
    
    # 4. Generate final predictions
    preds_open = preds_potential.numpy()
    preds_open[is_unknown.numpy()] = -1 # Samples exceeding their threshold are classified as unknown (-1)

    # --- Evaluation Report ---
    unique_labels = sorted(set(y_true_all.tolist()) | set(preds_open.tolist()))
    target_names = [le.inverse_transform([l])[0] if l != -1 else "Unknown" for l in unique_labels]
    
    print("\nOpen-set Evaluation Results (including unknown class = -1)")
    print(classification_report(y_true_all, preds_open, labels=unique_labels, target_names=target_names, zero_division=0))
    print("Confusion matrix:"); print(confusion_matrix(y_true_all, preds_open, labels=unique_labels))
    
    acc = accuracy_score(y_true_all, preds_open)
    pre = precision_score(y_true_all, preds_open, average="macro", zero_division=0)
    rec = recall_score(y_true_all, preds_open, average="macro", zero_division=0)
    f1  = f1_score(y_true_all, preds_open, average="macro", zero_division=0)
    print(f"\nOpen-set Overall -> Acc: {acc:.4f}, Prec: {pre:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    y_true_binary = (y_true_all != -1).astype(int); y_pred_binary = (preds_open != -1).astype(int)
    if len(np.unique(y_pred_binary)) > 1:
        # Handle cases where confusion matrix might not be 2x2
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        TPR = tp / (tp + fn) if (tp + fn) > 0 else 0 # Recall for known classes
        FPR = fp / (fp + tn) if (fp + tn) > 0 else 0 # False positive rate for unknown classes
        print(f"\nTPR (Known Class Recognition Rate): {TPR:.4f}"); print(f"FPR (Unknown Class Misclassification Rate): {FPR:.4f}")
    else:
        print("\nCannot calculate TPR/FPR because predictions only contain one class.")
    
    return model

# =========================
# 6. Visualization Function
# =========================
def visualize_feature_space(model, train_loader, test_loader, le, device, save_dir):
    print("\n--- Feature Space Visualization ---")
    train_features, train_labels = extract_features_global(model, train_loader, device)
    test_features, test_labels = extract_features_global(model, test_loader, device)
    
    train_features = train_features.numpy()
    train_labels = train_labels.numpy()
    test_features = test_features.numpy()
    test_labels = test_labels.numpy()

    if TSNE_MAX_SAMPLES is not None and train_features.shape[0] > TSNE_MAX_SAMPLES:
        print(f"Sampling training data from {train_features.shape[0]} down to {TSNE_MAX_SAMPLES} points...")
        indices = np.random.choice(train_features.shape[0], TSNE_MAX_SAMPLES, replace=False)
        train_features = train_features[indices]
        train_labels = train_labels[indices]

    if TSNE_MAX_SAMPLES is not None and test_features.shape[0] > TSNE_MAX_SAMPLES:
        print(f"Sampling test data from {test_features.shape[0]} down to {TSNE_MAX_SAMPLES} points...")
        indices = np.random.choice(test_features.shape[0], TSNE_MAX_SAMPLES, replace=False)
        test_features = test_features[indices]
        test_labels = test_labels[indices]
    
    all_features = np.vstack([train_features, test_features])
    all_labels = np.concatenate([train_labels, test_labels])
    source_labels = np.array(['train'] * len(train_labels) + ['test'] * len(test_labels))
    
    print(f"Running t-SNE on {all_features.shape[0]} samples...")
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, n_iter=1000, random_state=42, verbose=1)
    tsne_results = tsne.fit_transform(all_features)
    
    print("Plotting results...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 14))
    
    unique_labels = np.unique(all_labels)
    known_labels = sorted([l for l in unique_labels if l != -1])
    
    colors = plt.cm.get_cmap('tab20', len(known_labels))
    color_map = {label: colors(i) for i, label in enumerate(known_labels)}
    color_map[-1] = '#808080'

    for label in unique_labels:
        label_text = f'Class {le.inverse_transform([label])[0]}' if label != -1 else 'Unknown'
        train_mask = (all_labels == label) & (source_labels == 'train')
        if np.any(train_mask):
            ax.scatter(tsne_results[train_mask, 0], tsne_results[train_mask, 1],
                       c=[color_map[label]], marker='o', s=25, alpha=0.5,
                       label=f'Train: {label_text}')
        test_mask = (all_labels == label) & (source_labels == 'test')
        if np.any(test_mask):
            ax.scatter(tsne_results[test_mask, 0], tsne_results[test_mask, 1],
                       c=[color_map[label]], marker='x', s=60, alpha=0.9,
                       label=f'Test: {label_text}')

    ax.set_title(f't-SNE Visualization (Perplexity={TSNE_PERPLEXITY})', fontsize=20)
    handles, labels = ax.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax.legend(unique_legend.values(), unique_legend.keys(), bbox_to_anchor=(1.03, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    output_path = os.path.join(save_dir, "tsne_visualization_dynamic_threshold.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nDone! Visualization saved to: {output_path}")

# =========================
# 7. Main Workflow
# =========================
def main():
    # Check if data files exist
    if not os.path.exists(TRAIN_PKL) or not os.path.exists(TEST_PKL):
        print(f"Error: Please ensure training data '{TRAIN_PKL}' and test data '{TEST_PKL}' exist.")
        print("You might need to modify the TRAIN_PKL and TEST_PKL variables at the top of the script.")
        return

    train_df, test_df = pd.read_pickle(TRAIN_PKL), pd.read_pickle(TEST_PKL)
    
    if os.path.exists(LABEL_ENCODER_PATH):
        with open(LABEL_ENCODER_PATH, "rb") as f: le = pickle.load(f)
    else:
        le = LabelEncoder(); le.fit(train_df["label"].values)
        with open(LABEL_ENCODER_PATH, "wb") as f: pickle.dump(le, f)
    
    known_class = le.classes_
    X_train = [log_normalize(f) for f in train_df["features"]]
    X_test  = [log_normalize(f) for f in test_df["features"]]

    train_loader = DataLoader(FlowDataset(X_train, train_df["label"].values, le, known_class), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader  = DataLoader(FlowDataset(X_test, test_df["label"].values, le, known_class), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trained_model = train_eval_model(train_loader, test_loader, num_classes=len(le.classes_), le=le, device=device)
    visualize_feature_space(trained_model, train_loader, test_loader, le, device, SAVE_DIR)
    
    print("\nAll tasks completed.")


if __name__ == "__main__":
    main()