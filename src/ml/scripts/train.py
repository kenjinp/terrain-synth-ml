import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from ml.utils.save import save_checkpoint, load_checkpoint, save_some_examples
from ml.config import train as config
from ml.data.dataset import TerrainDataset
from ml.model.generator import Generator
from ml.model.discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

torch.backends.cudnn.benchmark = True


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        # with torch.cuda.amp.autocast():
        y_fake = gen(x)

        D_real = disc(y, y)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = disc(y, y_fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        # with torch.cuda.amp.autocast():
        D_fake = disc(y, y_fake)
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
        G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
            wandb.log({
                "D_real": torch.sigmoid(D_real).mean().item(),
                "D_fake": torch.sigmoid(D_fake).mean().item(),
                "G_fake_loss": G_fake_loss.item(),
                "G_loss": G_loss.item()
            })


def main():
    wandb.login()
    wandb.init(
        project=config.PROJECT_NAME,
    )
    disc = Discriminator(in_channels=1, initial_channels=2).to(config.DEVICE)
    gen = Generator(in_channels=3, out_channels=1,
                    features=64).to(config.DEVICE)
    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(
        gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    dataset = TerrainDataset(data_path=config.DATA_DIR)
    train_size = int(len(dataset) * 0.9)
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, test_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    def checkin(epoch):
        save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
        save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
        save_some_examples(gen, val_loader, epoch,
                           folder=config.EVALUATIONS_FOLDER, use_wandb=True)

    if config.SAVE_MODEL:
        checkin(-1)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )
        checkin(epoch)

    wandb.finish()


if __name__ == "__main__":
    main()
