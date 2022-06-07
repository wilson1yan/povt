from .vqvae import VQVAE
from .povt import POVT


def load_ckpt(ckpt_path, set_eval=True, **override_kwargs):
    import os.path as osp
    import torch
    import pickle
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args_path = osp.join(osp.dirname(osp.dirname(ckpt_path)), 'args')
    args = pickle.load(open(args_path, 'rb'))
    for k, v in override_kwargs.items():
        setattr(args, k, v)
    
    model = get_model(args, torch.device('cpu'))
    model.load_state_dict(ckpt['model'])

    if set_eval:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model


def get_model(args, device):
    if args.model == 'vqvae':
        model = VQVAE(args).to(device)
    elif args.model == 'povt':
        model = POVT(args).to(device)
    else:
        raise ValueError('Invalid model name:', args.model)
    return model
