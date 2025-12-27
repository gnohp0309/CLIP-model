import torch
from torch.utils.data import DataLoader
from torchvision import models
import config
from dataset import UITVIC_DATA, KTVIC_DATA
from model import CLIP, text_encoder_base
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter
import os


def main():
    print("--- Bắt đầu load dữ liệu UITVIC ---")
    train_dataset = UITVIC_DATA(
        json_path=config.UITVIC_TRAIN_JSON,
        img_dir=config.UITVIC_TRAIN_IMG,
        augment=True
    )

    train_dataset = Subset(train_dataset, range(0, 1000))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    print(f"--> Số lượng mẫu training: {len(train_dataset)}")

    print("--- Khởi tạo Model ---")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    resnet.fc = torch.nn.Identity()

    model = CLIP(vision_encoder=resnet, text_encoder=text_encoder_base)
    model = model.to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"--- Bắt đầu Train ({config.EPOCHS} Epochs) ---")
    writer = SummaryWriter('runs/experiment_1')
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0

        for batch_idx, (imgs, texts) in enumerate(train_loader):
            imgs = imgs.to(config.DEVICE)

            optimizer.zero_grad()
            loss = model(imgs, texts)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Training Loss', loss.item(), global_step)

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch + 1}/{config.EPOCHS}] | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"===> KẾT THÚC EPOCH {epoch + 1} | Avg Loss: {avg_loss:.4f}")

        save_path = f"weights/clip_vietnamese_ep{epoch + 1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"--> Đã lưu model tại: {save_path}")

    writer.close()


if __name__ == "__main__":
    main()