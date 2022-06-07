import os
import itertools
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .obj_attention import AttentionStack, LayerNorm, RightShift, \
    GlimpseRightShift, flatten, expand, AddBroadcastPosEmbed
from ..utils import inverse_masked_select, view_range, Quantizer, \
    quantize_bboxes, extract_glimpses
from ..dist_ops import is_master_process
from .base import Encoder

 
class POVT(nn.Module):
    has_bbox = True
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_timesteps_gen = args.n_timesteps_gen
        self.n_timesteps_train = args.n_cond_base + 1
        self.n_cond_base = args.n_cond_base
        self.n_cond_obj = args.n_cond_obj
        self.n_obj = self.K = args.n_obj
        assert self.n_cond_obj >= self.n_cond_base

        from . import load_ckpt
        self.vqvae = load_ckpt(args.vqvae_ckpt, set_eval=True)
        self.shapes = [(self.n_cond_obj + 1, self.n_obj, 5), 
                       (self.n_cond_base + 1, *self.vqvae.latent_shape), 
                       (self.n_cond_obj + 1, self.n_obj, *args.bbox_shape)]
        self.n_codes = [2, args.n_quantize_values, args.n_quantize_values, self.vqvae.n_codes[0]]
        self.input_dims = [2, args.n_quantize_values, args.n_quantize_values, 
                           self.vqvae.embedding_dim, args.glimpse_hiddens]
        self.seq_per_T = self.n_obj * 5 + np.prod(self.vqvae.latent_shape)

        self.glimpse_encoder = Encoder(args.glimpse_hiddens, args.glimpse_layers,
                                       [args.glimpse_downsample, args.glimpse_downsample])

        self.base_n = np.prod(self.vqvae.latent_shape)
        self.bbox_n = np.prod(args.bbox_shape)

        self.fc_ins = nn.ModuleList([
            nn.Linear(inp_dim, args.hidden_dim, bias=False)
            for inp_dim in self.input_dims
        ])
        [fc_in.weight.data.normal_(std=0.02) for fc_in in self.fc_ins]

        self.right_shift = RightShift(args.hidden_dim)
        self.g_right_shift = GlimpseRightShift(args.hidden_dim, args.bbox_shape)

        self.pos_embds = nn.ModuleList([
            AddBroadcastPosEmbed((1, *s), args.hidden_dim)
            for s in [
                (self.n_cond_obj + 1, self.n_obj, 5),
                (self.n_cond_base + 1, *self.vqvae.latent_shape),
                (self.n_cond_obj + 1, self.n_obj, *args.bbox_shape)
            ]
        ])
    
        self.attn_stack = AttentionStack(
            self.shapes, args.hidden_dim, args.heads, args.layers, args.dropout,
            args.attn_type, args.attn_dropout, causal=True
        )

        self.norms = nn.ModuleList([LayerNorm(args.hidden_dim)
                                    for _ in range(len(self.n_codes))]) 
        self.fc_outs = nn.ModuleList([nn.Linear(args.hidden_dim, n_code, bias=False)
                                      for n_code in self.n_codes])        
        [fc_out.weight.data.copy_(torch.zeros(n_code, args.hidden_dim))
         for fc_out, n_code in zip(self.fc_outs, self.n_codes)]

        self.shift_quantizer = Quantizer(-1, 1, args.n_quantize_values)
        self.scale_quantizer = Quantizer(0, 1, args.n_quantize_values)
        
        self.create_attn_mask()

    def create_attn_mask(self):
        block_n = np.prod(self.shapes[1][1:])
        mask = torch.ones(self.n_timesteps_train * block_n,
                          self.n_timesteps_train * block_n,
                          dtype=torch.bool)
        mask = torch.tril(mask)
        self.register_buffer('base_attn_mask', mask.float())
    
        bbox_n = np.prod(self.args.bbox_shape)
        block_n = bbox_n + 5
        gen_block = self.n_timesteps_train * block_n
        mask = torch.ones((self.n_cond_obj + 1) * block_n, (self.n_cond_obj + 1) * block_n, dtype=torch.bool)
        mask[:-gen_block, -gen_block:] = 0 
        mask[-gen_block:, -gen_block:] = torch.tril(mask[-gen_block:, -gen_block:])
        single_block = torch.block_diag(torch.ones(bbox_n, bbox_n, dtype=torch.bool),
                                        torch.zeros(5, 5, dtype=torch.bool))
        block_mask = torch.block_diag(*([single_block] * self.n_timesteps_train))
        mask[-gen_block:, -gen_block:] |= block_mask
        self.register_buffer('obj_ot_mask', mask.float())

        K = self.n_obj
        block_size = bbox_n
        mask = torch.zeros(K * (block_size + 5), K * (block_size + 5), dtype=torch.bool)
        for r in range(2 * K):
            for c in range(2 * K):
                start_r = r // 2 * (block_size + 5) + (r % 2) * block_size
                end_r = (r + 1) // 2 * (block_size + 5) + ((r + 1) % 2) * block_size

                start_c = c // 2 * (block_size + 5) + (c % 2) * block_size
                end_c = (c + 1) // 2 * (block_size + 5) + ((c + 1) % 2) * block_size

                if c % 2 == 0:
                    mask[start_r:end_r, start_c:end_c] = True
                elif r % 2 == 1:
                    if c < r:
                        mask[start_r:end_r, start_c:end_c] = True
                    elif r == c:
                        assert end_r - start_r == 5 and end_c - start_c == 5, f'{r}, {c}'
                        mask[start_r:end_r, start_c:end_c] = torch.tril(torch.ones(5, 5, dtype=torch.bool))

        mask = mask.unsqueeze(0).repeat_interleave(self.n_timesteps_train, dim=0)
        cond_mask = torch.ones(K * (block_size + 5), K * (block_size + 5), dtype=torch.bool)
        cond_mask = cond_mask.unsqueeze(0).repeat_interleave(self.n_cond_obj - self.n_cond_base, dim=0)
        mask = torch.cat((cond_mask, mask), dim=0)
        assert mask.shape[0] == self.n_cond_obj + 1

        self.register_buffer('obj_oo_mask', mask.float())

    def create_sampling_idxs(self, timesteps, debug=False):
        idxs = []
        for t in range(self.n_timesteps_train):
            for k in range(self.n_obj):
                idxs.append((0, t, k))
                idxs.extend([(1, t, k, i) for i in range(4)])
            tmp_idxs = list(itertools.product(*[range(s) for s in self.vqvae.latent_shape]))
            tmp_idxs = [(2, t, *i) for i in tmp_idxs]
            idxs.extend(tmp_idxs)
        assert len(idxs) == self.n_timesteps_train * self.seq_per_T

        if debug:
            return idxs

        rtn_idxs = []
        for t in range(timesteps):
            if t < self.n_timesteps_train:
                rtn_idxs.append(idxs[t * self.seq_per_T: (t + 1) * self.seq_per_T])
            else:
                rtn_idxs.append(idxs[-self.seq_per_T:])
        
        assert sum([len(i) for i in rtn_idxs]) == timesteps * self.seq_per_T
        
        return rtn_idxs
        
    @property
    def metrics(self):
        return ['loss']

    def clear_cache(self):
        self.attn_stack.clear_cache()

    def sample(self, n, cond_frames=None, cond_bboxes=None, cond_valid_bboxes=None, 
               fast_decode=True, return_bbox=False):
        # cond_frames: BTCHW
        if cond_frames is not None:
            cond_frames = cond_frames[:, :1]
            cond_bboxes = cond_bboxes[:, :1]
            cond_valid_bboxes = cond_valid_bboxes[:, :1]
        device = self.fc_ins[0].weight.device 

        samples = []
        video = []
        all_idxs = self.create_sampling_idxs(self.n_timesteps_gen)

        if cond_frames is not None:
            assert cond_frames.shape[0] == n
            # BT'K, BT'K4, BT'HW
            cond_encodings = self.encode_frames(cond_frames, cond_bboxes, cond_valid_bboxes)[1] # B*
            cond_encodings = torch.cat([
                torch.cat((cond_encodings[0].unsqueeze(-1),
                           cond_encodings[1]), dim=-1).flatten(start_dim=2), # BT'(K5)
                cond_encodings[2].flatten(start_dim=2) # BT'(HW)
            ], dim=-1).flatten(start_dim=1) # B(T'(K5+HW))
            cond_latents = cond_encodings.shape[1]
        else:
            cond_latents = 0
        
        n_cond_used = 0

        with torch.no_grad():
            if is_master_process():
                pbar = tqdm(total=self.seq_per_T * self.n_timesteps_gen)
            prev_idx = None
            for t in range(self.n_timesteps_gen):
                idxs = all_idxs[t]
                sample = [
                    torch.zeros(n, self.n_obj, dtype=torch.long).to(device),
                    torch.zeros(n, self.n_obj, 4, dtype=torch.long).to(device),
                    torch.zeros(n, *self.vqvae.latent_shape, dtype=torch.long).to(device)
                ]
                cur_v = torch.zeros(n, 3, self.args.resolution, self.args.resolution,
                                    dtype=torch.float32).to(device)

                for i, idx in enumerate(idxs): 
                    batch_idx = (slice(None, None), *idx[2:])
                    embeddings = [
                        F.one_hot(sample[0], 2).float(),
                        F.one_hot(sample[1], self.args.n_quantize_values).float(),
                        self.vqvae.dictionary_lookup(sample[2])
                    ]
                    if t < self.args.sequence_length:
                        decode_step = i + t * self.seq_per_T
                    else:
                        decode_step = i + (self.args.sequence_length - 1) * self.seq_per_T

                    if fast_decode:
                        if prev_idx is None:
                            prev_idx = idx

                        embeddings_slice = embeddings[prev_idx[0]][(slice(None, None), *prev_idx[2:])]
                        if i == 0:
                            if len(samples) > 0:
                                prev_frames = [
                                    [s[0] for s in samples[-self.n_cond_obj:]],
                                    [s[1] for s in samples[-self.n_cond_obj:]],
                                    [s[2] for s in samples[-self.n_cond_obj:]]
                                ]
                                prev_v = video[-self.n_cond_obj:]
                            else:
                                prev_frames = [[], [], []]
                                prev_v = []
                            n_obj_cond = len(prev_frames[0])
                            prev_frames[0].append(sample[0])
                            prev_frames[1].append(sample[1])
                            prev_frames[2].append(sample[2])
                            prev_frames = [torch.stack(f, dim=1) for f in prev_frames]
                            prev_frames = [
                                F.one_hot(prev_frames[0], 2).float(),
                                F.one_hot(prev_frames[1], self.args.n_quantize_values).float(),
                                self.vqvae.dictionary_lookup(prev_frames[2])
                            ]

                            prev_v.append(cur_v)
                            prev_v = torch.stack(prev_v, dim=1)
                            
                            if t == 0:
                                embeddings_slice = sample[prev_idx[0]][(slice(None, None), *prev_idx[2:])]
                            else:
                                embeddings_slice = samples[-1][prev_idx[0]][(slice(None, None), *prev_idx[2:])]

                            if prev_idx[0] == 0: # valid
                                embeddings_slice = F.one_hot(embeddings_slice, 2).float()
                            elif prev_idx[0] == 1: # bbox
                                embeddings_slice = F.one_hot(embeddings_slice, self.args.n_quantize_values).float()
                            else:
                                assert prev_idx[0] == 2
                                embeddings_slice = self.vqvae.dictionary_lookup(embeddings_slice)

                            has_cond = True
                            embeddings_slice = embeddings_slice.unsqueeze(1)
                            logits = self.step_sample(prev_v, (prev_frames, embeddings_slice),
                                               decode_step=decode_step, decode_idx=idx,
                                               has_cond=has_cond, prev_idx=prev_idx, n_obj_cond=n_obj_cond)
                        else:
                            has_cond = False
                            embeddings_slice = embeddings_slice.unsqueeze(1)
                            logits = self.step_sample(None, embeddings_slice,
                                            decode_step=decode_step, decode_idx=idx,
                                            has_cond=has_cond, prev_idx=prev_idx, n_obj_cond=n_obj_cond)
                        logits = logits.squeeze(1)
                    else:
                        if len(samples) > 0:
                            prev_frames = [
                                [s[0] for s in samples[-self.n_cond_obj:]],
                                [s[1] for s in samples[-self.n_cond_obj:]],
                                [s[2] for s in samples[-self.n_cond_obj:]]
                            ]
                            prev_frames = [torch.stack(f, dim=1) for f in prev_frames]

                            prev_frames = [
                                F.one_hot(prev_frames[0], 2).float(),
                                F.one_hot(prev_frames[1], self.args.n_quantize_values).float(),
                                self.vqvae.dictionary_lookup(prev_frames[2])
                            ] # BT_1KD, BT_1K4D, BT_1HWD

                            prev_v = video[-self.n_cond_obj:]
                        else:
                            prev_frames = None
                            prev_v = []

                        embeddings = [e.unsqueeze(1) for e in embeddings] # add time dim
                        if prev_frames is not None:
                            embeddings = [torch.cat((prev_e, e), dim=1)
                                          for e, prev_e in zip(embeddings, prev_frames)]
                        prev_v.append(cur_v)
                        v_inp = torch.stack(prev_v, dim=1)
                        logits = self.step(v_inp, embeddings)[idx[0]][(slice(None, None), *idx[1:])]

                    if n_cond_used < cond_latents:
                        sample[idx[0]][batch_idx] = cond_encodings[:, n_cond_used]
                        n_cond_used += 1
                    else:
                        probs = F.softmax(logits, dim=-1)
                        sample[idx[0]][batch_idx] = torch.multinomial(probs, 1).squeeze(-1)

                    if is_master_process():
                        pbar.update(1)

                    prev_idx = idx
                
                samples.append(sample)
                v = self.vqvae.decode(sample[2])
                v = torch.clamp(v, 0, 1)
                video.append(v)
            samples = list(zip(*samples))
            samples = [torch.stack(s, dim=1) for s in samples]
            video = torch.stack(video, dim=1)
            if is_master_process():
                pbar.close()
        self.clear_cache()

        if return_bbox:
            return video, (samples[0], samples[1])
        return video # BCTHW in [0, 1]


    def encode_frames(self, x, cond_bboxes, cond_valid_bboxes):
        # BTCHW
        self.vqvae.eval()

        bboxes, bboxes_labels  = quantize_bboxes(cond_bboxes, 
                                                 self.shift_quantizer, 
                                                 self.scale_quantizer) 
        bboxes_one_hot = F.one_hot(bboxes_labels, self.args.n_quantize_values).float()

        if x is not None:
            B, T = x.shape[:2]
            embeddings, encodings = self.vqvae.encode(x.flatten(end_dim=1))
            reconstructions = self.vqvae.decode(encodings)
            reconstructions = view_range(reconstructions, 0, 1, (B, T))

            embeddings = view_range(embeddings, 0, 1, (B, T))
            encodings = view_range(encodings, 0, 1, (B, T))
        else:
            embeddings, encodings, reconstructions = None, None, None

        valid_bboxes = cond_valid_bboxes.long()
        valid_bboxes_one_hot = F.one_hot(valid_bboxes, 2).float()
        
        return (valid_bboxes_one_hot, bboxes_one_hot, embeddings), \
               (valid_bboxes, bboxes_labels, encodings), reconstructions

    def _compute_aux_info(self, valid_bboxes, bboxes):
        # x: *1D, *4D
        valid_bboxes = torch.argmax(valid_bboxes, dim=-1).bool() # *1
        shift, scale = bboxes.chunk(2, dim=-2)
        shift = torch.argmax(shift, dim=-1) # *2
        scale = torch.argmax(scale, dim=-1) # *2
        shift = self.shift_quantizer.dequantize(shift)
        scale = self.scale_quantizer.dequantize(scale)
        bboxes = torch.cat((shift, scale), dim=-1) # *4
        return valid_bboxes, bboxes

    def step_sample(self, video, x, decode_step, decode_idx, prev_idx, has_cond, n_obj_cond):
        attn_mask = [self.base_attn_mask, self.obj_ot_mask, self.obj_oo_mask]

        if has_cond:
            cond, x = x
            B, T1 = cond[0].shape[:2]
            
            valid_bboxes, bboxes = self._compute_aux_info(cond[0], cond[1])
            glimpses = extract_glimpses(video.flatten(end_dim=1),
                                        bboxes.flatten(end_dim=1),
                                        valid_bboxes.flatten(end_dim=1),
                                        glimpse_size=(self.args.glimpse_size, self.args.glimpse_size),
                                        return_padded=False)
            glimpse_embeddings = self.glimpse_encoder(glimpses)
            if glimpse_embeddings.shape[2:] != self.args.bbox_shape[0]:
                glimpse_embeddings = F.interpolate(glimpse_embeddings, size=self.args.bbox_shape)
            glimpse_embeddings = inverse_masked_select(glimpse_embeddings, valid_bboxes.flatten(end_dim=1))
            glimpse_embeddings = glimpse_embeddings.movedim(2, -1)
            glimpse_embeddings = view_range(glimpse_embeddings, 0, 1, (B, T1)) # BT_1KGGD

            glimpse_embeddings = self.fc_ins[-1](glimpse_embeddings)
            glimpse_embeddings = self.g_right_shift(glimpse_embeddings, None) # always shift
            pos_embd = self.pos_embds[2].construct_pos_embd() # 11TKGGD
            glimpse_embeddings = glimpse_embeddings.unsqueeze(1)
            glimpse_embeddings = (glimpse_embeddings + pos_embd[:, :, :T1]).squeeze(1)

            cond = [cond[0], *cond[1].chunk(2, dim=-2), cond[2][:, -self.n_timesteps_train:]]
            cond = [fc_in(cond_i) for fc_in, cond_i in zip(self.fc_ins, cond)]
            cond = [torch.cat((cond[0].unsqueeze(3), cond[1], cond[2]), dim=-2), cond[3]]
            cond = [cond_i[:, :-1] for cond_i in cond] # remove last / current timestep
            
        if prev_idx[0] == 0:
            in_i = 0
        elif prev_idx[0] == 1:
            in_i = 1 if prev_idx[-1] < 2 else 2
        else:
            in_i = 3
        
        h = self.fc_ins[in_i](x) 

        if has_cond:
            cond_bboxes, cond_embeddings = cond # BT_1K5D, BT_1HWD
            cond_bboxes1, cond_bboxes2 = cond_bboxes[:, :-self.n_cond_base], cond_bboxes[:, -self.n_cond_base:]
            cond_bboxes2 = cond_bboxes2.flatten(start_dim=2, end_dim=3)[:, -self.n_timesteps_train:]
            T2 = cond_bboxes2.shape[1]
            cond_embeddings = cond_embeddings.flatten(start_dim=2, end_dim=3)
            cond_flat = torch.cat((cond_bboxes2, cond_embeddings), dim=2)
            cond_flat = cond_flat.flatten(start_dim=1, end_dim=2) # B(T_2(K5+HW))D
            h = torch.cat((cond_flat, h), dim=1)
            h = self.right_shift(h, 0)

            cond_flat, h = h[:, :-1], h[:, [-1]]
            cond_flat = view_range(cond_flat, 1, 2, (T2, self.K * 5 + self.base_n)) # BT_2(K5+HW)D
            cond_bboxes2, cond_embeddings = torch.split(cond_flat, (self.K * 5, self.base_n), dim=2)
            cond_bboxes2 = view_range(cond_bboxes2, 2, 3, (self.K, 5))
            cond_bboxes = torch.cat((cond_bboxes1, cond_bboxes2), dim=1)
            cond_embeddings = view_range(cond_embeddings, 2, 3, self.vqvae.latent_shape)

            cond_bboxes = self.pos_embds[0](cond_bboxes.unsqueeze(1)).squeeze(1)
            cond_embeddings = self.pos_embds[1](cond_embeddings.unsqueeze(1)).squeeze(1)

            cond = (cond_bboxes, cond_embeddings)
        else:
            h = self.right_shift(h, decode_step)

        shape_i, decode_idx = decode_idx[0], decode_idx[1:]
        if shape_i == 0:
            h = h.unsqueeze(1)
            pos_embd_i = 0
            pos_embd_idx = (n_obj_cond, decode_idx[1], 0)
        elif shape_i == 1:
            pos_embd_i = 0
            pos_embd_idx = (n_obj_cond, decode_idx[1], decode_idx[2] + 1)
        else:
            pos_embd_i = 1
            pos_embd_idx = decode_idx
        h = view_range(h, 1, 2, (1,) * len(pos_embd_idx))
        h = self.pos_embds[pos_embd_i](h.unsqueeze(1), decode_step, (0, *pos_embd_idx))
        h = h.flatten(start_dim=1, end_dim=-2)

        aux_info = dict()
        aux_info['decode_idx'] = (shape_i, *decode_idx)
        aux_info['has_cond'] = has_cond

        if has_cond:
            h, tensor_stats = flatten((*cond, glimpse_embeddings, h))
            aux_info['tensor_stats'] = tensor_stats
        
        h = self.attn_stack(h, aux_info, decode_step, attn_mask)

        if has_cond:
            h = expand(h, tensor_stats)
            h = h[-1]  # B1D
        
        if shape_i == 0:
            out_i = 0
        elif shape_i == 1:
            out_i = 1 if decode_idx[-1] < 2 else 2
        else:
            out_i = 3
        
        h = self.norms[out_i](h)
        logits = self.fc_outs[out_i](h)
        return logits 


    def step(self, video, x, targets=None, decode_step=None, decode_idx=None): 
        # video: BT_1CHW, x: (BT_1KD, BT_1K4D, BT_1HWD)
        B, T1, K = x[0].shape[:3]
        T_mask = min(T1, self.n_timesteps_train)
        base_n = np.prod(self.shapes[1][1:])
        bbox_n = np.prod(self.args.bbox_shape)

        attn_mask = [self.base_attn_mask[:T_mask * base_n, :T_mask * base_n], 
                     self.obj_ot_mask[-T1 * (bbox_n + 5):, -T1 * (bbox_n + 5):], 
                     self.obj_oo_mask[-T1:]]
        
        valid_bboxes, bboxes = self._compute_aux_info(x[0], x[1])

        glimpses = extract_glimpses(video.flatten(end_dim=1),
                                    bboxes.flatten(end_dim=1),
                                    valid_bboxes.flatten(end_dim=1),
                                    glimpse_size=(self.args.glimpse_size, self.args.glimpse_size),
                                    return_padded=False)
        glimpse_embeddings = self.glimpse_encoder(glimpses)
        if glimpse_embeddings.shape[2:] != self.args.bbox_shape[0]:
            glimpse_embeddings = F.interpolate(glimpse_embeddings, size=self.args.bbox_shape)
        glimpse_embeddings = inverse_masked_select(glimpse_embeddings, valid_bboxes.flatten(end_dim=1))
        glimpse_embeddings = glimpse_embeddings.movedim(2, -1)
        glimpse_embeddings = view_range(glimpse_embeddings, 0, 1, (B, T1)) # BT_1KGGD

        x = [x[0], *x[1].chunk(2, dim=-2), x[2][:, -self.n_timesteps_train:]]
        x = [fc_in(x_i) for fc_in, x_i in zip(self.fc_ins, x)]
        x = [torch.cat((x[0].unsqueeze(3), x[1], x[2]), dim=-2), x[3]] # BTK5D, BTHWD
        glimpse_embeddings = self.fc_ins[-1](glimpse_embeddings)

        h_cond, h = list(zip(*[(x_i[:, :-self.n_timesteps_train], 
                                x_i[:, -self.n_timesteps_train:]) for x_i in x]))
        h = self.right_shift(h, decode_step)
        h = [torch.cat((h_cond_i, h_i), dim=1) for h_cond_i, h_i in zip(h_cond, h)]

        glimpse_embeddings = self.g_right_shift(glimpse_embeddings, decode_step)

        h = [*h, glimpse_embeddings]
        h = [pos_embd(h_i.unsqueeze(1), None, None).squeeze(1)
             for pos_embd, h_i in zip(self.pos_embds, h)]
        h, tensor_stats = flatten(h)
        aux_info = dict(tensor_stats=tensor_stats)

        h = self.attn_stack(h, aux_info, decode_step, attn_mask=attn_mask)

        h = expand(h, tensor_stats)[:-1] # throw away glimpse embeddings

        h = [h_i[:, -self.n_timesteps_train:] for h_i in h] # BT_2K5D, BT_2HWD
        
        h = [*torch.split(h[0], [1, 2, 2], dim=3), h[1]]
        h[0] = h[0].squeeze(3)
        h = [norm(h_i) for norm, h_i in zip(self.norms, h)]
        logits = [fc_out(h_i) for fc_out, h_i in zip(self.fc_outs, h)]
        logits = [logits[0], torch.cat((logits[1], logits[2]), dim=3), logits[3]]
        if targets is not None:
            assert len(logits) == len(targets)
            losses = [F.cross_entropy(logit.movedim(-1, 1), target,
                                      reduction='none', ignore_index=-1)
                      for logit, target in zip(logits, targets)]
            loss = torch.cat([l.flatten() for l in losses])
            loss = loss.sum() / (loss > 0).sum()
            return loss

        return logits

    def forward(self, x, bboxes, valid_bboxes):
        # x: BTCHW
        B, T = x.shape[:2]
        # embeddings: (BTKD, BTK4D, BTHWD), encodings: (BTK, BTK4, BTHW)
        embeddings, encodings, recons = self.encode_frames(x, bboxes, valid_bboxes) 

        n_obj_cond = np.random.randint(0, self.n_cond_obj - self.n_cond_base + 1)
        idx = np.random.randint(n_obj_cond, T - self.n_timesteps_train)

        embeddings = [e[:, idx - n_obj_cond:idx + self.n_timesteps_train] for e in embeddings]
        encodings = [e[:, idx:idx + self.n_timesteps_train] for e in encodings]
        recons = recons[:, idx - n_obj_cond:idx + self.n_timesteps_train]
        
        loss = self.step(recons, embeddings, encodings)
        return dict(loss=loss)
        
