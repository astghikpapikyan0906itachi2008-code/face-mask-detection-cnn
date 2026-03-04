from pathlib import Path


ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"


IMAGE_SIZE = (224, 224)
CHANNELS = 3

BATCH_SIZE = 32
EPOCHS = 50          
LEARNING_RATE = 1e-4
RANDOM_STATE = 42


THRESHOLD = 0.6
