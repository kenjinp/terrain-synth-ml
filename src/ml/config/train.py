from ml.utils.device import get_device
import torchvision.transforms as T
import torch

PROJECT_NAME = "Terrain-Painter-pix2pix"
DEVICE = get_device()
DATA_DIR = ".data/archive"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500
LOAD_MODEL = True
SAVE_MODEL = True
EVALUATIONS_FOLDER = ".evaluations"
CHECKPOINT_DISC = ".checkpoints/disc.pth.tar"
CHECKPOINT_GEN = ".checkpoints/gen.pth.tar"

both_transform = T.Compose(
    [
        # T.Lambda(lambda image: T.functional.convert_image_dtype(image)),
        # T.Lambda(lambda image: (image / 256.0)),
        T.Resize((256, 256)),
    ]
)

transform_only_input = T.Compose(
    [
        T.Lambda(lambda image: (image * 1/256)),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[
            0.5, 0.5, 0.5]),
    ]
)

transform_only_mask = T.Compose(
    [
        T.Lambda(lambda image: (image * 1/65535)),
        T.Normalize(mean=[0.5], std=[0.5]),
        T.Grayscale(num_output_channels=1),
    ]
)


def denormalize(image): return image * 0.5 + 0.5
