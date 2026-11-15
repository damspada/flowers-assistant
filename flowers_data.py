# Data pipeline for the Flowers 102 project.
# Everyone imports from here: same splits, class names and preprocessing.
# From a notebook in notebooks/:
#   import sys
#   sys.path.append("..")
#   from flowers_data import get_split, load_image, to_tensor, CLASS_NAMES

from pathlib import Path
import random

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torchvision import transforms
from torchvision.datasets import Flowers102

SEED = 42

# data/ is always next to this file
DATA_DIR = Path(__file__).resolve().parent / "data"

# input size of the classifiers
IMAGE_SIZE = 224

# ImageNet normalization, used by the pretrained models
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# class names, index in the list = label (0..101)
# checked with the image counts on the official page:
# https://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html
CLASS_NAMES = [
    'pink primrose', 'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',
    'english marigold', 'tiger lily', 'moon orchid', 'bird of paradise', 'monkshood',
    'globe thistle', 'snapdragon', "colt's foot", 'king protea', 'spear thistle',
    'yellow iris', 'globe-flower', 'purple coneflower', 'peruvian lily', 'balloon flower',
    'giant white arum lily', 'fire lily', 'pincushion flower', 'fritillary', 'red ginger',
    'grape hyacinth', 'corn poppy', 'prince of wales feathers', 'stemless gentian',
    'artichoke', 'sweet william', 'carnation', 'garden phlox', 'love in the mist',
    'mexican aster', 'alpine sea holly', 'ruby-lipped cattleya', 'cape flower',
    'great masterwort', 'siam tulip', 'lenten rose', 'barbeton daisy', 'daffodil',
    'sword lily', 'poinsettia', 'bolero deep blue', 'wallflower', 'marigold', 'buttercup',
    'oxeye daisy', 'common dandelion', 'petunia', 'wild pansy', 'primula', 'sunflower',
    'pelargonium', 'bishop of llandaff', 'gaura', 'geranium', 'orange dahlia',
    'pink-yellow dahlia?', 'cautleya spicata', 'japanese anemone', 'black-eyed susan',
    'silverbush', 'californian poppy', 'osteospermum', 'spring crocus', 'bearded iris',
    'windflower', 'tree poppy', 'gazania', 'azalea', 'water lily', 'rose', 'thorn apple',
    'morning glory', 'passion flower', 'lotus', 'toad lily', 'anthurium', 'frangipani',
    'clematis', 'hibiscus', 'columbine', 'desert-rose', 'tree mallow', 'magnolia',
    'cyclamen', 'watercress', 'canna lily', 'hippeastrum', 'bee balm', 'ball moss',
    'foxglove', 'bougainvillea', 'camellia', 'mallow', 'mexican petunia', 'bromelia',
    'blanket flower', 'trumpet creeper', 'blackberry lily',
]
NUM_CLASSES = len(CLASS_NAMES)

# split names inside setid.mat
SPLIT_KEYS = {"train": "trnid", "val": "valid", "test": "tstid"}


# fix all the seeds, so every run gives the same results
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# download the official files (images, labels, splits), only the first time
def download():
    Flowers102(root=DATA_DIR, split="train", download=True)


# image paths and labels (0..101) of "train", "val" or "test"
def get_split(split):
    download()
    folder = DATA_DIR / "flowers-102"

    # labels in the file go from 1 to 102
    labels_file = loadmat(folder / "imagelabels.mat", squeeze_me=True)
    all_labels = labels_file["labels"] - 1

    # ids of the images in this split
    split_file = loadmat(folder / "setid.mat", squeeze_me=True)
    ids = split_file[SPLIT_KEYS[split]]

    paths = []
    for i in ids:
        file_name = f"image_{i:05d}.jpg"     # e.g. image_00042.jpg
        paths.append(folder / "jpg" / file_name)

    # ids start from 1, the array from 0
    labels = all_labels[ids - 1]
    return paths, labels


# open an image (path or PIL image) as RGB
def load_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    img = Image.open(image)
    return img.convert("RGB")


# preprocessing for every classifier input: dataset, uploaded and generated images
# resize keeps the proportions, then 224x224 crop in the center
# training augmentation goes on top of this
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# image -> tensor (1, 3, 224, 224), ready for a classifier
def to_tensor(image):
    img = load_image(image)
    x = preprocess(img)
    x = x.unsqueeze(0)      # batch dimension
    return x
