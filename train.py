import torch
import wandb
import tqdm
from torch.utils.data import DataLoader

from configs.config import cfg
from data.dataset import CarlaVeriDataset
from data.transforms import train_transforms, inference_transforms
from data.samplers import PKSampler
from models.clip_senet import CLIP_SENet
from losses.loss_wrapper import LossWrapper
from utils.metrics import evaluate_reid

def main():
    wandb.init(project="CLIP-SENet-ReID", name="V2_Normalized_Run")

    # 1. Load Data
    train_dataset = CarlaVeriDataset(cfg.train_image_dir, transform=train_transforms)
    cfg.num_classes = len(train_dataset.unique_pids)
    
    pk_sampler = PKSampler(train_dataset, p=4, k=8)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=pk_sampler, num_workers=cfg.num_workers)
    
    query_dataset = CarlaVeriDataset(cfg.query_image_dir, transform=inference_transforms)
    gallery_dataset = CarlaVeriDataset(cfg.gallery_image_dir, transform=inference_transforms)
    query_loader = DataLoader(query_dataset, batch_size=128, num_workers=4)
    gallery_loader = DataLoader(gallery_dataset, batch_size=128, num_workers=4)

    
    model = CLIP_SENet(cfg).to(cfg.device)
    criterion = LossWrapper(num_classes=cfg.num_classes).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_map = 0

    
    for epoch in range(cfg.epochs):
        model.train()
        loop = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}")
        
        for images, labels, _, _, _ in loop:
            images, labels = images.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()

            features, logits = model(images, labels)
            loss, ce, sc = criterion(logits, features, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            loop.set_postfix(loss=loss.item())
            wandb.log({"train/loss": loss.item()})

        
        mAP, rank1, rank5, rank10 = evaluate_reid(model, query_loader, gallery_loader)
        print(f"\nEpoch {epoch+1} | mAP: {mAP:.4f} | Rank-1: {rank1:.4f}")
        wandb.log({"val/mAP": mAP, "val/rank1": rank1})

        if mAP > best_map:
            best_map = mAP
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved Best Model!")

if __name__ == "__main__":
    main()