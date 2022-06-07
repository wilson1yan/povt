import torch.nn as nn
import torch.nn.functional as F

from .base import SamePadConv2d, Encoder, Decoder, Codebook


class VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.n_codes = (args.n_codes,)

        self.encoder = Encoder(args.n_hiddens, args.n_res_layers, args.downsample)
        self.decoder = Decoder(args.n_hiddens, args.n_res_layers, args.downsample, 3)

        self.pre_vq_conv = SamePadConv2d(args.n_hiddens, args.embedding_dim, 1)
        self.post_vq_conv = SamePadConv2d(args.embedding_dim, args.n_hiddens, 1)

        self.codebook = Codebook(args.n_codes, args.embedding_dim)
    
    @property
    def metrics(self):
        return ['loss', 'recon_loss', 'commitment_loss', 'perplexity']

    @property
    def latent_shape(self):
        ds = self.args.downsample
        input_shape = (self.args.resolution, self.args.resolution)
        return tuple([s // d for s, d in zip(input_shape, ds)])

    def encode(self, x):
        x = self.encoder(x)
        h = self.pre_vq_conv(x)
        vq_output = self.codebook(h)
        return vq_output['embeddings'].movedim(1, -1), vq_output['encodings']

    def decode(self, encodings):
        h = F.embedding(encodings, self.codebook.embeddings)
        h = self.post_vq_conv(h.movedim(-1, 1))
        return self.decoder(h)

    def dictionary_lookup(self, x):
        return self.codebook.dictionary_lookup(x)

    def forward(self, x):
        # x: B1CHW, masks: B1KHW
        x = x.flatten(end_dim=1)
        h = self.encoder(x)
        z = self.pre_vq_conv(h)
        vq_output = self.codebook(z)
        x_recon = self.decoder(self.post_vq_conv(vq_output['embeddings']))
        recon_loss = F.mse_loss(x_recon, x) / 0.06

        return dict(loss=recon_loss + vq_output['commitment_loss'],
                    recon_loss=recon_loss,
                    commitment_loss=vq_output['commitment_loss'],
                    perplexity=vq_output['perplexity'],
                    original=x, reconstruction=x_recon)
