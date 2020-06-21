from .unet.model import Unet


unet_cfg = {
    'encoder_depth': 5,
    'decoder_use_batchnorm': True,
    'decoder_channels': (512, 256, 128, 64, 32),
    'decoder_attention_type': 'scse',
    'in_channels': 3,
}


def generate_unet(num_classes, encoder_name='resnet34'):
    return Unet(classes=num_classes, encoder_name=encoder_name, **unet_cfg)

