import model_converter
from clip import CLIP
from decoder import VAE_Decoder
from diffusion import Diffusion
from encoder import VAE_Encoder


def preload_models_from_standard_weights(ckpt_path, device):
    '''
    想試試看把這四個models用pytorch vision的model重寫一遍
    '''
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    encoder = VAE_Encoder().to(device)
    encoder.load_state_dict(state_dict['encoder'], strict=True)

    decoder = VAE_Decoder().to(device)
    decoder.load_state_dict(state_dict['decoder'], strict=True)

    diffusion = Diffusion().to(device)
    diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'encoder': encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }