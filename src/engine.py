import time
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Automatic Mixed Precision for accelerated GPU training
        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds) * 100
    epoch_f1 = f1_score(all_targets, all_preds, average="macro") * 100
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    for images, targets, _ in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    eval_loss = running_loss / len(dataloader.dataset)
    eval_acc = accuracy_score(all_targets, all_preds) * 100
    eval_f1 = f1_score(all_targets, all_preds, average="macro") * 100
    return eval_loss, eval_acc, eval_f1, np.array(all_probs), np.array(all_preds), np.array(all_targets)


def fit_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_path="best_model.pth"):
    scaler = torch.amp.GradScaler("cuda" if torch.cuda.is_available() else "cpu")
    best_val_f1 = 0.0
    history = {"train_loss": [], "train_acc": [], "train_f1": [], "val_loss": [], "val_acc": [], "val_f1": []}

    print(f"[TRAINING] Starting {num_epochs} Epochs on Device: {device}")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc, val_f1, _, _, _ = evaluate(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(f"Epoch [{epoch:02d}/{num_epochs:02d}] "
              f"Train Loss: {tr_loss:.4f} | Train Acc: {tr_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val Macro-F1: {val_f1:.2f}%")

        # Save Best Model Checkpoint based on Validation Macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_f1": best_val_f1
            }, save_path)
            print(f"  [SAVED] New best model checkpoint saved to: {save_path} (Val F1: {best_val_f1:.2f}%)")

    total_time = (time.time() - start_time) / 60
    print(f"[COMPLETE] Training completed in {total_time:.2f} minutes. Best Val F1: {best_val_f1:.2f}%")
    return history