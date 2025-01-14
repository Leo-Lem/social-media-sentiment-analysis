from argparse import ArgumentParser
from os import path, makedirs

parser = ArgumentParser(
    description="Train a model to predict the opinion of a text on climate change")

parser.add_argument("model", type=str, choices=["baseline", "blank", "sentiment"],
                    help="Model to train")

parser.add_argument("--sample", action="store_true",
                    help="Use a sample dataset")
parser.add_argument("--epochs", type=int, default=10,
                    help="Number of epochs to train the model")
parser.add_argument("--batch", type=int, default=32,
                    help="Batch size for training")
parser.add_argument("--data", type=str, default="res",
                    help="Path to the data directory")
args = parser.parse_args()

BASE_PATH = path.dirname(__file__)
DATA_PATH = path.join(BASE_PATH, args.data)
OUT_PATH = path.join(BASE_PATH, ".out")
makedirs(OUT_PATH, exist_ok=True)

MODEL = args.model
EPOCHS = args.epochs
BATCH_SIZE = args.batch
SAMPLE = args.sample
