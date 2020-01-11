from .waveunet.training.train_gan import train as waveunet_train_gan
from .waveunet.training.train_mse import train as waveunet_train_mse
from .waveunet.training.train_feat_loss import train as waveunet_train_fl
from .spectral_u_net.train import train as spectral_unet_train
from .spectral_u_net.train_fl import train as spectral_unet_fl
from .acoustic_scenes.train import train as scene_net_train
from .acoustic_scenes_spectral.train import train as scene_net_spectral_train

TASKS = {
    "waveunet_train_gan": waveunet_train_gan,
    "waveunet_train_mse": waveunet_train_mse,
    "waveunet_train_fl": waveunet_train_fl,
    "spectral_unet_train": spectral_unet_train,
    "spectral_unet_fl": spectral_unet_fl,
    "scene_net_train": scene_net_train,
    "scene_net_spectral_train": scene_net_spectral_train,
}
