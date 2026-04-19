import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.utils import save_image

print(torch.__version__)
print(torch.cuda.is_available())


# # =====================================================
# # CONFIG (CPU)
# # =====================================================
# DEVICE = "cpu"
# BATCH_SIZE = 2
# EPOCHS = 20
# LR = 1e-4
# IMG_SIZE = 256   # resize so batching works
#
# TRAIN_DIR = "data/train/degraded"
# VAL_DIR   = "data/val/degraded"
#
# OUT_DIR = "outputs_selfsup"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# # =====================================================
# # DATASET (SAFE)
# # =====================================================
# class DegradedDataset(Dataset):
#     def __init__(self, root):
#         assert os.path.exists(root), f"Path not found: {root}"
#
#         self.files = [
#             os.path.join(root, f)
#             for f in os.listdir(root)
#             if f.lower().endswith((".png", ".jpg", ".jpeg"))
#         ]
#
#         assert len(self.files) > 0, f"No images found in {root}"
#
#         self.t = T.Compose([
#             T.Resize((IMG_SIZE, IMG_SIZE)),
#             T.ToTensor()
#         ])
#
#     def __len__(self):
#         return len(self.files)
#
#     def __getitem__(self, idx):
#         img = Image.open(self.files[idx]).convert("RGB")
#         return self.t(img)
#
# # =====================================================
# # DEGRADATION MODEL (SELF-SUPERVISION)
# # =====================================================
# class DegradationModel(nn.Module):
#     def forward(self, x):
#         if random.random() < 0.7:
#             x = F.avg_pool2d(x, 2)
#             x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
#
#         noise = torch.randn_like(x) * random.uniform(0.02, 0.06)
#         return torch.clamp(x + noise, 0, 1)
#
# # =====================================================
# # RESTORATION MODEL
# # =====================================================
# class ResidualBlock(nn.Module):
#     def __init__(self, c):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(c, c, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(c, c, 3, padding=1)
#         )
#
#     def forward(self, x):
#         return x + self.block(x)
#
# class SelfSupRestorer(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.head = nn.Conv2d(3, 64, 3, padding=1)
#         self.body = nn.Sequential(*[ResidualBlock(64) for _ in range(8)])
#         self.tail = nn.Conv2d(64, 3, 3, padding=1)
#
#     def forward(self, x):
#         x = self.head(x)
#         x = self.body(x)
#         x = self.tail(x)
#         return torch.clamp(x, 0, 1)
#
# # =====================================================
# # LOSSES
# # =====================================================
# def edge_loss(x, y):
#     return (
#         F.l1_loss(x[:, :, :, 1:], y[:, :, :, 1:]) +
#         F.l1_loss(x[:, :, 1:, :], y[:, :, 1:, :])
#     )
#
# def color_loss(x, y):
#     return F.l1_loss(x.mean([2, 3]), y.mean([2, 3]))
#
# # =====================================================
# # TRAIN
# # =====================================================
# def train():
#     print("🔍 Loading datasets...")
#
#     train_ds = DegradedDataset(TRAIN_DIR)
#     val_ds   = DegradedDataset(VAL_DIR)
#
#     print(f"✅ Train images: {len(train_ds)}")
#     print(f"✅ Val images:   {len(val_ds)}")
#
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
#     val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)
#
#     model = SelfSupRestorer().to(DEVICE)
#     degrade = DegradationModel()
#     opt = torch.optim.Adam(model.parameters(), lr=LR)
#
#     for epoch in range(EPOCHS):
#         model.train()
#         train_loss = 0
#
#         for img in train_loader:
#             img = img.to(DEVICE)
#
#             restored = model(img)
#             redeg = degrade(restored)
#
#             l_cons = F.l1_loss(redeg, img)
#             l_edge = edge_loss(restored, img)
#             l_col  = color_loss(restored, img)
#
#             loss = l_cons + 0.2*l_edge + 0.1*l_col
#
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#
#             train_loss += loss.item()
#
#         # ---------------- VALIDATION ----------------
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for img in val_loader:
#                 img = img.to(DEVICE)
#                 val_loss += F.l1_loss(degrade(model(img)), img).item()
#
#         val_loss /= len(val_loader)
#
#         print(
#             f"Epoch {epoch+1}/{EPOCHS} | "
#             f"Train loss: {train_loss:.4f} | "
#             f"Val loss: {val_loss:.4f}"
#         )
#
#         # save 1 sample
#         save_image(restored[:1], f"{OUT_DIR}/epoch_{epoch+1}.png")
#
#     torch.save(model.state_dict(), f"{OUT_DIR}/model.pth")
#     print("✅ Training finished")
#
# # =====================================================
# if __name__ == "__main__":
#     train()
