from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from detectron2.config import LazyCall as L
from torch.utils.data.distributed import DistributedSampler

from data import ImageFileTrain, DataGenerator

#Dataloader
train_dataset = DataGenerator(
    data = ImageFileTrain(
        alpha_dir='path/to/alpha',
        fg_dir='path/to/alpha',
        bg_dir='path/to/alpha',
        root='path/to/Adobe_Image_Matting'
    ),
    phase = 'train'
)

dataloader = OmegaConf.create()
dataloader.train = L(DataLoader)(
    dataset = train_dataset,
    batch_size=15,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    sampler=L(DistributedSampler)(
        dataset = train_dataset,
    ),
    drop_last=True
)