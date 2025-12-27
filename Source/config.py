import torch

UITVIC_TRAIN_IMG = 'Dataset/uitvic_dataset/coco_uitvic_train/coco_uitvic_train'
UITVIC_TRAIN_JSON = 'Dataset/uitvic_dataset/uitvic_captions_train2017.json'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
MAX_LENGTH = 50

TEXT_EMBED_DIM = 768
IMAGE_EMBED_DIM = 2048
PROJECTION_DIM = 256

print(f"--> Cấu hình: Thiết bị đang dùng là {DEVICE}")