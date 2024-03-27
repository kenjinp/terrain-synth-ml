import torch
from ml.config import train as config
import wandb
from torchvision.utils import save_image
import os


def save_some_examples(gen, val_loader, epoch, folder, use_wandb=False):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = config.denormalize(y_fake)
        x = config.denormalize(x)
        if use_wandb:
            y_sample = y_fake.permute(0, 2, 3, 1)
            y_sample = y_fake.detach().cpu().numpy()
            y_sample = wandb.Image(
                y_sample, caption="Generated Output")
            x_sample = x.permute(0, 2, 3, 1)
            x_sample = x_sample.detach().cpu().numpy()
            x_sample = wandb.Image(
                x_sample, caption="Input")
            wandb.log({"y": y_sample, "x": x_sample})
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        save_image(x, folder + f"/input_{epoch}.png")
        if epoch == 1:
            save_image(config.denormalize(y), folder + f"/label_{epoch}.png")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")

    if not os.path.exists(checkpoint_file):
        print("-- Failed to load checkpoint, missing file")
        return

    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
