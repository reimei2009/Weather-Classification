import os
import re
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from config import device, output_dir

def get_latest_epoch(prefix="model_epoch_", suffix=".pth"):
    max_epoch = 0
    latest_model = None
    for filename in os.listdir(output_dir):
        match = re.match(fr"{prefix}(\d+){suffix}", filename)
        if match:
            epoch_num = int(match.group(1))
            if epoch_num > max_epoch:
                max_epoch = epoch_num
                latest_model = filename
    return max_epoch, latest_model

@torch.no_grad()
def evaluate(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    for images, labels in tqdm(data_loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    avg_loss = total_loss / len(data_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    return {'loss': avg_loss, 'accuracy': acc, 'precision': precision, 'recall': recall}

def train_one_epoch(model, train_loader, optimizer, device, scaler):
    model.train()
    train_losses = []
    correct_train = 0
    total_train = 0
    loop = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_losses.append(loss.item())
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)
        train_accuracy = correct_train / total_train
        loop.set_postfix(train_loss=loss.item(), train_accuracy=train_accuracy)
    return np.mean(train_losses), train_accuracy

def save_best_models(model, val_result, epoch, loss_save_path, acc_save_path, best_val_loss, best_val_accuracy):
    updated = False
    if val_result['loss'] < best_val_loss:
        best_val_loss = val_result['loss']
        torch.save(model.state_dict(), loss_save_path)
        print(f"✅Best loss model saved at epoch {epoch + 1:03d}")
        updated = True
    if val_result['accuracy'] > best_val_accuracy:
        best_val_accuracy = val_result['accuracy']
        torch.save(model.state_dict(), acc_save_path)
        print(f"✅Best accuracy model saved at epoch {epoch + 1:03d}")
        updated = True
    return updated, best_val_loss, best_val_accuracy

def save_epoch_model(model, epoch, prefix="model_epoch_"):
    path = os.path.join(output_dir, f"{prefix}{epoch + 1:03d}.pth")
    torch.save(model.state_dict(), path)
    print(f"Model for epoch {epoch + 1:03d} saved at {path}")

def fit(epochs, lr, model, train_loader, test_loader, device, start_epoch=0, step_size=1, gamma=0.1,
        loss_save_path='best_loss_model.pth', acc_save_path='best_acc_model.pth', patience=5, prefix="model_epoch_"):
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scaler = GradScaler()
    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    early_stop_counter = 0
    lr_no_improve_counter = 0
    for epoch in range(start_epoch, start_epoch + epochs):
        start_time = time.time()
        print('=' * 25 + f' Epoch {epoch + 1:03d}/{start_epoch + epochs} ' + '=' * 25)
        train_loss_avg, train_accuracy = train_one_epoch(model, train_loader, optimizer, device, scaler)
        train_result = evaluate(model, train_loader, device)
        train_result['train_loss'] = train_loss_avg
        train_result['train_accuracy'] = train_accuracy
        history.append(train_result)
        val_result = evaluate(model, test_loader, device)
        updated, best_val_loss, best_val_accuracy = save_best_models(
            model, val_result, epoch, loss_save_path, acc_save_path, best_val_loss, best_val_accuracy
        )
        if not updated:
            early_stop_counter += 1
            lr_no_improve_counter += 1
            print(f'🚫🚫Stopping counter {early_stop_counter}/{patience}')
            print(f'🚫🚫Learning rate counter {lr_no_improve_counter}/{3}')
        else:
            early_stop_counter = 0
            lr_no_improve_counter = 0
        save_epoch_model(model, epoch, prefix)
        minutes, seconds = divmod(time.time() - start_time, 60)
        print(f"Epoch time: {int(minutes):02d}:{int(seconds):02d} (mm:ss)")
        print(f"Epoch [{epoch+1}/{start_epoch + epochs}] "
              f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_accuracy:.4f} | "
              f"Val Loss: {val_result['loss']:.4f} | Val Acc: {val_result['accuracy']:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
        if early_stop_counter >= patience:
            print("❌❌Early stopping triggered")
            break
        if lr_no_improve_counter >= 3:
            scheduler.step()
            print(f"🆎🆎Learning rate reduced to {scheduler.get_last_lr()[0]:.6f} at epoch {epoch + 1:03d}")
            lr_no_improve_counter = 0
    return history