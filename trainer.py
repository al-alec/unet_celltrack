import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

def estimate_pos_weight(loader, max_batches=50, device='cpu'):
    pos = 0.0
    neg = 0.0
    for i, (_, m) in enumerate(loader):
        if i >= max_batches:
            break
        m = m.float().to(device)
        pos += m.sum().item()
        neg += (1.0 - m).sum().item()
    w = neg / max(pos, 1.0)
    return torch.tensor([w], dtype=torch.float32, device=device)

def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.contiguous().view(pred.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)

    intersection = (pred * target).sum(dim=1)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)))

    return loss.mean()


def bce_dice_loss(pred, target, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dsc = dice_loss(pred, target)

    return bce * bce_weight + dsc * (1 - bce_weight)


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, pos_weight=None):
        super().__init__()
        self.bce_weight = bce_weight
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=self.pos_weight)
        dsc = dice_loss(pred, target)

        return bce * self.bce_weight + dsc * (1 - self.bce_weight)


class UnetTrainer:
    def __init__(self, model, train_loader, val_loader=None, lr=1e-3, device='cuda', patience=5, log_dir='runs'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

        pos_weight = estimate_pos_weight(self.train_loader, max_batches=50, device=self.device)
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = BCEDiceLoss(bce_weight=0.5, pos_weight=pos_weight)
        self.history = {'train_loss': [], 'val_loss': []}

        # Early stopping
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None

        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0



    def train_epoch(self):
        self.model.train()
        total_loss = 0
        accumulation_steps = 4

        for idx, (images, masks) in enumerate(self.train_loader):
            if idx % 10 == 0:
                print(f"image idx {idx}")
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks) / accumulation_steps
            loss.backward()
            if (idx + 1) % accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

            # Log batch loss to TensorBoard
            self.writer.add_scalar('Loss/train_batch', loss.item() * accumulation_steps, self.global_step)
            self.global_step += 1

            torch.cuda.empty_cache()

        return total_loss / len(self.train_loader)

    def validate(self):
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)

            # Log epoch train loss to TensorBoard
            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)

            val_loss = self.validate()
            if val_loss is not None:
                self.history['val_loss'].append(val_loss)

                # Log epoch val loss to TensorBoard
                self.writer.add_scalar('Loss/val_epoch', val_loss, epoch)

                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

                # Early stopping check
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.counter = 0
                    self.best_model_state = self.model.state_dict().copy()
                    print(f"  → New best model! Val Loss improved to {val_loss:.4f}")
                else:
                    self.counter += 1
                    print(f"  → No improvement. Counter: {self.counter}/{self.patience}")

                    if self.counter >= self.patience:
                        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                        print(f"Restoring best model with val_loss={self.best_loss:.4f}")
                        self.model.load_state_dict(self.best_model_state)
                        break
            else:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

        self.writer.close()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
