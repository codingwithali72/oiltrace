import os
import sys
import argparse
import time
import csv
import torch
import torch.optim as optim
from pathlib import Path

# Add the ml-service folder (or parent) to sys.path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import model, losses, metrics, dataset (handling both 'ml-service' and 'ml_service')
try:
    from ml_service.model.deeplabv3 import create_model
    from ml_service.model.losses import get_loss_function
    from ml_service.model.metrics import compute_metrics
    from ml_service.model.dataset import OilSpillDataset
except ModuleNotFoundError:
    # Fallback to importing directly if executed as script
    # This works if sys.path has ml-service itself
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from model.deeplabv3 import create_model
    from model.losses import get_loss_function
    from model.metrics import compute_metrics
    from model.dataset import OilSpillDataset

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='auto', help="cuda or cpu or auto")
    parser.add_argument('--amp', action='store_true', help="Use mixed precision (CUDA only)")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--resume', type=str, default=None, help="Path to checkpoint")
    parser.add_argument('--dry-run', action='store_true', help="Run 1 dummy epoch for testing")
    parser.add_argument('--data-root', type=str, default="", help="Prefix for dataset paths")
    parser.add_argument('--checkpoint-dir', type=str, default="ml-service/checkpoints")
    return parser.parse_args()

def save_checkpoint(path, model, optimizer, epoch, metrics_dict):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics_dict
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint to {path}")

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Resumed from {path} (epoch {epoch})")
    return epoch

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, size=10):
        self.size = size
    def __len__(self): return self.size
    def __getitem__(self, idx):
        # returns 2-channel input and 1-channel target
        return torch.randn(2, 512, 512), torch.randint(0, 2, (1, 512, 512)).float()

def main():
    args = get_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # AMP setup
    use_amp = args.amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Model
    model = create_model().to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Loss
    criterion = get_loss_function(name="combined").to(device)
    
    start_epoch = 1
    best_dice = -1.0
    
    if args.resume and os.path.exists(args.resume):
        start_epoch = load_checkpoint(args.resume, model, optimizer) + 1
        
    # Dataset and Dataloaders
    if args.dry_run:
        train_loader = torch.utils.data.DataLoader(DummyDataset(size=4), batch_size=2) # 2 batches of size 2
        val_loader = torch.utils.data.DataLoader(DummyDataset(size=2), batch_size=2)   # 1 batch of size 2
        epochs = 1
        print("--- DRY RUN MODE ---")
    else:
        # Load manifests (assume they exist)
        train_manifest = []
        with open("ml-service/data_manifests/train_manifest.csv") as f:
            train_manifest = list(csv.DictReader(f))
        
        val_manifest = []
        with open("ml-service/data_manifests/val_manifest.csv") as f:
            val_manifest = list(csv.DictReader(f))
            
        train_ds = OilSpillDataset(train_manifest, mode="train", root_dir=args.data_root)
        val_ds = OilSpillDataset(val_manifest, mode="val", root_dir=args.data_root)
        
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=4, shuffle=False)
        epochs = args.epochs
        
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_file = os.path.join(args.checkpoint_dir, 'training_log.csv')
    
    # Write header if new
    if not os.path.exists(log_file):
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'dice', 'iou', 'precision', 'recall', 'epoch_time_seconds'])

    for epoch in range(start_epoch, start_epoch + epochs):
        epoch_start = time.time()
        
        # Train pass
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Val pass
        model.eval()
        val_loss = 0.0
        val_metrics = {'dice': 0.0, 'iou': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                metrics = compute_metrics(outputs, targets)
                for k in val_metrics:
                    val_metrics[k] += metrics[k]
                    
        val_loss /= len(val_loader)
        for k in val_metrics:
            val_metrics[k] /= len(val_loader)
            
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Dice: {val_metrics['dice']:.4f} | IoU: {val_metrics['iou']:.4f} | "
              f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f} | Time: {epoch_time:.2f}s")
        
        # Logging
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, val_metrics['dice'], 
                             val_metrics['iou'], val_metrics['precision'], val_metrics['recall'], epoch_time])
                             
        # Checkpointing
        save_checkpoint(os.path.join(args.checkpoint_dir, 'last.pt'), model, optimizer, epoch, val_metrics)
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            save_checkpoint(os.path.join(args.checkpoint_dir, 'best.pt'), model, optimizer, epoch, val_metrics)
            
    if args.dry_run:
        print("\n--- Testing Reload ---")
        # Ensure reload works
        ckpt_path = os.path.join(args.checkpoint_dir, 'last.pt')
        load_checkpoint(ckpt_path, model, optimizer)
        print("Dry run end-to-end verified successfully.")

if __name__ == '__main__':
    main()
