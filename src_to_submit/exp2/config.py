NET = "vit"
PATCH_SIZE = 4
EPOCHS = 100
LR = 1e-3


# Use project-level dataset directory: E:/hw/deep_learning/data
DATA_ROOT = "../../data/"
NUM_WORKERS = 2

# Optimizer/loss options:
# OPTIMIZER: "adamw" or "sgd"
# LOSS_FN: "ce" or "ce_ls"
OPTIMIZER = "adamw"
LOSS_FN = "ce"
WEIGHT_DECAY = 0.05
MOMENTUM = 0.9
LABEL_SMOOTHING = 0.0
BATCH_SIZE = 32
IMAGE_SIZE = 64
USE_COSINE = False