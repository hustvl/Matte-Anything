from .common.train import train
from .common.model import model
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier

model.backbone.embed_dim = 768
model.backbone.num_heads = 12
model.decoder.in_chans = 768

train.max_iter = int(43100 / 10 / 2 * 100)
train.checkpointer.period = int(43100 / 10 / 2 * 10)

optimizer.lr=5e-4
lr_multiplier.scheduler.values=[1.0, 0.1, 0.05]
lr_multiplier.scheduler.milestones=[int(43100 / 10 / 2 * 30), int(43100 / 10 / 2 * 90)]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

train.init_checkpoint = './pretrained/mae_vit_b_fna.pth'
train.output_dir = './output_of_train/ViTMatte_B_100ep'