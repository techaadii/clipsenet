import torch
import wandb
import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast  # For 2x faster training

from configs.config import cfg
from data.dataset import CarlaVeriDataset
from data.transforms import train_transforms, inference_transforms
from data.samplers import PKSampler
from models.clip_senet import CLIP_SENet
from losses.loss_wrapper import LossWrapper
from utils.metrics import evaluate_reid


global_step = 0

def main():
    global global_step

    wandb.init(
        project="CLIP-SENet-ReID",
        name="V2_Normalized_AMP_Run"
    )

    # W&B Metric Tracking
    wandb.define_metric("train/global_step")
    wandb.define_metric("train/*", step_metric="train/global_step")
    wandb.define_metric("val/epoch")
    wandb.define_metric("val/*", step_metric="val/epoch")

    # ---------------- DATA ----------------
    print("Loading dataset...")
    train_dataset = CarlaVeriDataset(cfg.train_image_dir, transform=train_transforms)
    cfg.num_classes = len(train_dataset.unique_pids)
    
    # Ensure Batch Size is cleanly divisible by P and K (e.g., P=8, K=8 -> Batch=64)
    pk_sampler = PKSampler(train_dataset, p=8, k=8)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size, 
        sampler=pk_sampler, 
        num_workers=cfg.num_workers,
        pin_memory=True 
    )
    
    query_dataset = CarlaVeriDataset(cfg.query_image_dir, transform=inference_transforms)
    gallery_dataset = CarlaVeriDataset(cfg.gallery_image_dir, transform=inference_transforms)
    
    query_loader = DataLoader(query_dataset, batch_size=128, num_workers=4, pin_memory=True)
    gallery_loader = DataLoader(gallery_dataset, batch_size=128, num_workers=4, pin_memory=True)

    
    print("Initializing Model...")
    model = CLIP_SENet(cfg).to(cfg.device)
    criterion = LossWrapper(num_classes=cfg.num_classes).to(cfg.device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.learning_rate, 
        weight_decay=cfg.weight_decay
    )

    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

    
    scaler = GradScaler()

    wandb.watch(model, log="all", log_freq=100)
    best_map = 0.0

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(cfg.epochs):
        model.train()
        
        total_loss, total_ce, total_sc = 0, 0, 0
        num_batches = 0

        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [LR: {scheduler.get_last_lr()[0]:.1e}]")

        for images, labels, _, _, _ in loop:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()

            
            with autocast():
                # Model now only takes images during forward pass
                features, logits = model(images) 
                loss, ce, sc = criterion(logits, features, labels)

            # Backward pass with Scaler
            scaler.scale(loss).backward()
            
            # Unscale before clipping gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            
            # Step scaler and update
            scaler.step(optimizer)
            scaler.update()

            # Tracking
            total_loss += loss.item()
            total_ce += ce.item()
            total_sc += sc.item()
            num_batches += 1

            wandb.log({
                "train/global_step": global_step,
                "train/loss": loss.item(),
                "train/ce": ce.item(),
                "train/sc": sc.item(),
                "train/lr": scheduler.get_last_lr()[0]
            })

            global_step += 1
            loop.set_postfix(loss=f"{loss.item():.3f}", ce=f"{ce.item():.3f}", sc=f"{sc.item():.3f}")

        # Step the LR Scheduler at the end of the epoch
        scheduler.step()

        # -------- EPOCH SUMMARY --------
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Avg Loss: {total_loss/num_batches:.4f} | Avg CE: {total_ce/num_batches:.4f} | Avg SC: {total_sc/num_batches:.4f}")

       
        # Evaluate every 5 epochs to save time, and on the last 10 epochs
        if (epoch + 1) % 5 == 0 or epoch > cfg.epochs - 10:
            mAP, rank1, rank5, rank10 = evaluate_reid(model, query_loader, gallery_loader, device=cfg.device)

            print(f"\nValidation Metrics:")
            print(f"mAP:     {mAP:.4f}")
            print(f"Rank-1:  {rank1:.4f}")

            wandb.log({
                "val/epoch": epoch + 1,
                "val/mAP": mAP,
                "val/rank1": rank1,
                "val/rank5": rank5,
                "val/rank10": rank10
            })

            torch.save(model.state_dict(), "latest_model.pth")

            if mAP > best_map:
                best_map = mAP
                torch.save(model.state_dict(), "best_model.pth")
                print("Saved Best Model ✔")

    wandb.finish()

if __name__ == "__main__":
    main()