import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import view_range, flatten_idx, tensor_slice

class AddBroadcastPosEmbed(nn.Module):
    def __init__(self, shape, embd_dim, dim=-1):
        super().__init__()
        assert dim in [-1, 1] # only first or last dim supported
        self.shape = shape
        self.n_dim = n_dim = len(shape)
        self.embd_dim = embd_dim
        self.dim = dim

        chunk_sizes = [embd_dim // n_dim + (i < (embd_dim % n_dim))
                       for i in range(n_dim)]
        assert sum(chunk_sizes) == embd_dim, f'sum({chunk_sizes}) = {sum(chunk_sizes)} != {embd_dim}'
        self.emb = nn.ParameterDict({
             f'd_{i}': nn.Parameter(torch.randn(shape[i], chunk_sizes[i]) * 0.01
                                    if dim == -1 else
                                    torch.randn(chunk_sizes[i], shape[i]) * 0.01)
             for i in range(n_dim)
        })
    
    def construct_pos_embd(self):
        embs = []
        for i in range(self.n_dim):
            e = self.emb[f'd_{i}']
            if self.dim == -1:
                # (1, 1, ..., 1, self.shape[i], 1, ..., -1)
                e = e.view(1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)), -1)
                e = e.expand(1, *self.shape, -1)
            else:
                e = e.view(1, -1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)))
                e = e.expand(1, -1, *self.shape)
            embs.append(e)

        embs = torch.cat(embs, dim=self.dim)
        return embs
        
    def forward(self, x, decode_step=None, decode_idx=None):
        embs = self.construct_pos_embd()
        if decode_step is not None:
            embs = tensor_slice(embs, [0, *decode_idx, 0],
                                [x.shape[0], *(1,) * self.n_dim, x.shape[-1]])
        else:
            if x.shape[2] != embs.shape[2]:
                embs = embs[:, :, :x.shape[2]] # time embeddings as offsets for current t
        embs = embs.type_as(x)

        return x + embs


class RightShift(nn.Module):
    def __init__(self, embd_dim):
        super().__init__()
        self.embd_dim = embd_dim
        self.sos = nn.Parameter(torch.FloatTensor(embd_dim).normal_(std=0.02), requires_grad=True)

    def forward(self, x, decode_step):
        # x: (BTK5D, BTHWD) or B11...1D
        
        if decode_step is not None and decode_step > 0:
            return x

        if decode_step is None:
            T, K = x[0].shape[1:3]
            H, W = x[1].shape[2:4]
            x = [x_i.flatten(start_dim=2, end_dim=3) for x_i in x] # BT(K5)D, BT(HW)D
            x = torch.cat(x, dim=2) # BT(K5+HW)D
            x = x.flatten(start_dim=1, end_dim=2) # B(T(K5+HW))D
        else:
            old_shape = x.shape
            x = x.flatten(start_dim=1, end_dim=-2)
        
        sos = torch.ones(x.shape[0], 1, self.embd_dim, dtype=torch.float32).to(self.sos) * self.sos
        sos = sos.type_as(x)
        x = torch.cat([sos, x[:, :-1, :]], dim=1)

        if decode_step is None:
            x = view_range(x, 1, 2, (T, -1)) # BT(K5+HW)D
            x = torch.split(x, [K * 5, H * W], dim=2) # BT(K5)D, BT(HW)D
            x = [view_range(x_i, 2, 3, s)
                 for x_i, s in zip(x, [(K, 5), (H, W)])] # BTK5D, BTHWD
        else:
            x = x.view(old_shape)

        return x


class GlimpseRightShift(nn.Module):
    def __init__(self, embd_dim, bbox_shape):
        super().__init__()
        self.embd_dim = embd_dim
        self.bbox_shape = bbox_shape
        self.sos = nn.Parameter(torch.randn(*bbox_shape, embd_dim) * 0.02, requires_grad=True)
    
    def forward(self, x, decode_step):
        if decode_step is not None:
            if decode_step > 0:
                return x
            else:
                B, K = x.shape[:2]
                x = self.sos.unsqueeze(0).unsqueeze(0) # 11GGD
                x = x.repeat(B, K, 1, 1, 1) # BKGGD
                return x
        
        # BTKGGD
        T, K = x.shape[1], x.shape[2]
        sos = self.sos.view(1, 1, 1, *self.bbox_shape, self.embd_dim) # 111GGD
        sos = sos.repeat(x.shape[0], 1, K, 1, 1, 1) # B1KGGD
        sos = sos.type_as(x)
        x = torch.cat([sos, x[:, :-1]], dim=1) # BTKGGD
        return x
            
def flatten(x):
    # BT_1K5D, BT_2HWD, BT_1KGGD
    x_shapes = [x_i.shape[1:-1] for x_i in x]
    x = [x_i.flatten(start_dim=1, end_dim=-2) for x_i in x]
    x = torch.cat(x, dim=1)
    return x, x_shapes

def expand(x, x_shapes, has_head=False):
    dim = 2 if has_head else 1
    x = torch.split(x, [np.prod(s) for s in x_shapes], dim=dim)
    x = [view_range(x_i, dim, dim + 1, s) for x_i, s in zip(x, x_shapes)]
    return x
    

class AttentionStack(nn.Module):
    def __init__(
        self, shape, embd_dim, n_head, n_layer, dropout, attn_type, 
        attn_dropout, causal
    ):
        super().__init__()
        self.attn_nets = nn.ModuleList(
            [
                AttentionBlock(
                    shape=shape,
                    embd_dim=embd_dim,
                    n_head=n_head,
                    n_layer=n_layer,
                    dropout=dropout,
                    attn_type=attn_type,
                    attn_dropout=attn_dropout,
                    causal=causal,
                )
                for i in range(n_layer)
            ]
        )

    def clear_cache(self):
        [attn_net.clear_cache() for attn_net in self.attn_nets]

    def forward(self, x, aux_info, decode_step, attn_mask):
        """
        Args
        ------
            x: (b, d1, d2, ..., dn, embd_dim)
            cond: a dictionary of conditioning tensors

            (below is used only when sampling for fast decoding)
            decode: the enumerated rasterscan order of the current idx being sampled
            decode_step: a tuple representing the current idx being sampled
        """
        for net in self.attn_nets:
            x = net(x, aux_info, decode_step, attn_mask)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, shape, embd_dim, n_head, n_layer, dropout,
                 attn_type, attn_dropout, causal):
        super().__init__()

        self.pre_attn_norm = LayerNorm(embd_dim)
        self.post_attn_dp = nn.Dropout(dropout)
        self.attn = ObjectMultiHeadAttention(shape, embd_dim, embd_dim, n_head,
                                             n_layer, causal=causal, attn_type=attn_type,
                                             attn_kwargs=dict(attn_dropout=attn_dropout))
        
        self.pre_fc_norm = LayerNorm(embd_dim)
        self.post_fc_dp = nn.Dropout(dropout)
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=embd_dim, out_features=embd_dim * 4),
            GeLU2(),
            nn.Linear(in_features=embd_dim * 4, out_features=embd_dim),
        )
    
    def clear_cache(self):
        self.attn.clear_cache()

    def forward(self, x, aux_info, decode_step, attn_mask):
        h = self.pre_attn_norm(x)
        h = self.attn(h, aux_info, decode_step, attn_mask)
        h = self.post_attn_dp(h)
        x = x + h

        h = self.pre_fc_norm(x)
        h = self.fc_block(h)
        h = self.post_fc_dp(h)
        x = x + h

        return x

        
class ObjectMultiHeadAttention(nn.Module):
    attn_masks = dict()
    
    def __init__(self, shape, dim_q, dim_kv, n_head, n_layer,
                 causal, attn_type, attn_kwargs, proj_dim=None):
        super().__init__()
        # T_1K5, T_2HW, T_1KGG
        self.obj_shape, self.base_shape, self.glimpse_shape = shape
        self.T_obj, self.K = self.obj_shape[:2]
        self.T_base = self.base_shape[0]
        self.bbox_shape = self.glimpse_shape[2:]

        self.bbox_n = np.prod(self.bbox_shape)
        self.base_n = np.prod(self.base_shape[1:])

        if proj_dim is None:
            self.d_k = dim_q // n_head
            self.d_v = dim_kv // n_head
        else:
            self.d_k = self.d_v = proj_dim
        self.n_head = n_head

        self.w_qs = nn.Linear(dim_q, n_head * self.d_k, bias=False) # q
        self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim_q))

        self.w_ks = nn.Linear(dim_kv, n_head * self.d_k, bias=False) # k
        self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.w_vs = nn.Linear(dim_kv, n_head * self.d_v, bias=False) # v
        self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.fc = nn.Linear(n_head * self.d_v, dim_q, bias=True) # c
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))

        self.base_attn = FullAttention(None, causal, **attn_kwargs)
        self.full_attn = FullAttention(shape=None, causal=False, attn_dropout=0.)

        self.cache = None
 
    def clear_cache(self):
        self.cache = None

    def perform_attn(self, q, k, v, attn, decode_step, flatten=False, attn_mask=None):
        # q, k, v: B(n_head)*D
        if flatten:
            old_shape = q.shape[2:-1]
            q = q.flatten(start_dim=2, end_dim=-2)
            k = k.flatten(start_dim=2, end_dim=-2)
            v = v.flatten(start_dim=2, end_dim=-2)

        a = attn(q, k, v, decode_step, attn_mask)
        a = a.movedim(1, -2).flatten(start_dim=-2)
        a = self.fc(a)

        if flatten:
            a = view_range(a, 1, 2, old_shape)
        return a

    def _combine_obj(self, glimpses, bboxes):
        # glimpses: B(n_head)TKGGD, bboxes: B(n_head)TK5D
        glimpses = glimpses.flatten(start_dim=4, end_dim=5) # B(n_head)TK(GG)D
        obj = torch.cat((glimpses, bboxes), dim=4) # B(n_head)TK(GG+5)D
        return obj
    
    def forward(self, x, aux_info, decode_step=None, attn_mask=None):
        base_attn_mask, obj_ot_mask, obj_oo_mask = attn_mask

        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = view_range(self.w_qs(x), -1, None, (n_head, d_k))
        k = view_range(self.w_ks(x), -1, None, (n_head, d_k))
        v = view_range(self.w_vs(x), -1, None, (n_head, d_v))

        # (b, seq_len, n_head, d) -> (b, n_head, seq_len, d)
        q = q.movedim(-2, 1)
        k = k.movedim(-2, 1)
        v = v.movedim(-2, 1)

        if decode_step is not None:
            has_cond = aux_info['has_cond']
            if has_cond:            
                B = q.shape[0]
                self.cache = dict(
                    k_base=torch.zeros((B, n_head, *self.base_shape, self.d_k), dtype=q.dtype).to(x.device),
                    v_base=torch.zeros((B, n_head, *self.base_shape, self.d_v), dtype=q.dtype).to(x.device),
                    k_bbox=torch.zeros((B, n_head, self.K, 5, self.d_k), dtype=q.dtype).to(x.device),
                    v_bbox=torch.zeros((B, n_head, self.K, 5, self.d_v), dtype=q.dtype).to(x.device),
                )
            
            decode_idx = aux_info['decode_idx']
            T2 = decode_idx[1]
            
            if has_cond:
                tensor_stats = aux_info['tensor_stats']
                # B(n_head)T_1K5D, B(n_head)T_2HWD, B(n_head)(T_1+1)KGGD, B(n_head)1D
                bboxes_q, base_q, glimpses_q, q = expand(q, tensor_stats, has_head=True)
                bboxes_k, base_k, glimpses_k, k = expand(k, tensor_stats, has_head=True)
                bboxes_v, base_v, glimpses_v, v = expand(v, tensor_stats, has_head=True)
                assert base_q.shape[2] == T2

                # B(n_head)(T_1+1)K(GG+5)D
                obj_q = self._combine_obj(glimpses_q, F.pad(bboxes_q, (0, 0, 0, 0, 0, 0, 0, 1)))
                obj_k = self._combine_obj(glimpses_k, F.pad(bboxes_k, (0, 0, 0, 0, 0, 0, 0, 1)))
                obj_v = self._combine_obj(glimpses_v, F.pad(bboxes_v, (0, 0, 0, 0, 0, 0, 0, 1)))
                T1, K, GGp5 = obj_q.shape[2:5]

                self.cache['cond_k_bboxes'] = bboxes_k
                self.cache['cond_k_glimpses'] = glimpses_k
                self.cache['cond_v_bboxes'] = bboxes_v
                self.cache['cond_v_glimpses'] = glimpses_v

                self.cache['k_base'][:, :, :T2] = base_k
                self.cache['v_base'][:, :, :T2] = base_v

                p_bb = self.perform_attn(base_q, base_k, base_v, self.base_attn,
                                         None, flatten=True, 
                                         attn_mask=base_attn_mask[:T2 * self.base_n, :T2 * self.base_n])
                 
                p_bo = self.perform_attn(base_q.flatten(start_dim=3, end_dim=4),
                                         obj_k[:, :, -T2-1:-1].flatten(start_dim=3, end_dim=4),
                                         obj_v[:, :, -T2-1:-1].flatten(start_dim=3, end_dim=4),
                                         self.full_attn, None) # BT_2(HW)D
                p_bo = view_range(p_bo, 2, 3, (base_q.shape[3], base_q.shape[4])) # BT_2HWD
                p_base = (p_bb + p_bo) / 2 # BT_2HWD

                # obj-obj over time
                ot_q = obj_q.movedim(2, 3).flatten(start_dim=3, end_dim=4) # B(n_head)K(T_1(GG+5))D
                ot_k = obj_k.movedim(2, 3).flatten(start_dim=3, end_dim=4)
                ot_v = obj_v.movedim(2, 3).flatten(start_dim=3, end_dim=4)

                p_ot = self.perform_attn(ot_q, ot_k, ot_v, self.full_attn,
                                         decode_step=None, 
                                         attn_mask=obj_ot_mask[-T1 * (self.bbox_n + 5):, -T1 * (self.bbox_n + 5):])
                p_ot = view_range(p_ot, 2, 3, (T1, GGp5)) # BT_1(GG+5)D
                p_ot = p_ot.movedim(2, 1) # BT_1K(GG+5)D
 
                # obj-obj over obj
                ok_q = obj_q.flatten(start_dim=3, end_dim=4) # B(n_head)T_1(K(GG+5))D
                ok_k = obj_k.flatten(start_dim=3, end_dim=4)
                ok_v = obj_v.flatten(start_dim=3, end_dim=4)

                # BT(K(GG+5))D
                p_ok = self.perform_attn(ok_q, ok_k, ok_v, self.full_attn,
                                        decode_step=None, 
                                        attn_mask=obj_oo_mask[-T1:])
                p_ok = view_range(p_ok, 2, 3, (K, GGp5)) # BT_1K(GG+5)D
                
                p_oo = (p_ot + p_ok) / 2 # BT_1K(GG+5)D
                p_glimpses, p_bboxes = torch.split(p_oo, [GGp5 - 5, 5], dim=3)
                p_glimpses = view_range(p_glimpses, 3, 4, self.bbox_shape) # BT_1KGGD
                p_bboxes = p_bboxes[:, :-1] 

                cond = (p_bboxes, p_base, p_glimpses)

            shape_i, decode_idx = decode_idx[0], decode_idx[1:]
            if shape_i == 0: # valid
                k_i = decode_idx[1] 
                self.cache['k_bbox'][:, :, k_i, [0]] = k
                self.cache['v_bbox'][:, :, k_i, [0]] = v
            elif shape_i == 1: # bbox
                k_i, i = decode_idx[1:]
                self.cache['k_bbox'][:, :, k_i, [i+1]] = k
                self.cache['v_bbox'][:, :, k_i, [i+1]] = v
            else: # base
                t, h, w = decode_idx
                self.cache['k_base'][:, :, t, h, [w]] = k
                self.cache['v_base'][:, :, t, h, [w]] = v

            # B(n_head)T_1K5D, B(n_head)T_1KGGD
            bboxes_k = torch.cat((self.cache['cond_k_bboxes'], 
                                  self.cache['k_bbox'].unsqueeze(2)), dim=2) 
            glimpses_k = self.cache['cond_k_glimpses']
            obj_k = self._combine_obj(glimpses_k, bboxes_k) # B(n_head)T_1K(GG+5)D

            bboxes_v = torch.cat((self.cache['cond_v_bboxes'],
                                  self.cache['v_bbox'].unsqueeze(2)), dim=2)
            glimpses_v = self.cache['cond_v_glimpses']
            obj_v = self._combine_obj(glimpses_v, bboxes_v)

            if shape_i == 0 or shape_i == 1: # valid, bbox
                k_i = decode_idx[1]
 
                # ot attn
                ot_k = obj_k[:, :, :, k_i].flatten(2, 3) # B(n_head)(T_1(GG+5))D
                ot_v = obj_v[:, :, :, k_i].flatten(2, 3)
                
                t = obj_k.shape[2]
                m = obj_ot_mask[-(self.bbox_n + 5):, -t * (self.bbox_n + 5):]
                i = self.bbox_n
                if shape_i == 1:
                    i += 1 + decode_idx[2]
                p_ot = self.perform_attn(q, ot_k, ot_v, self.full_attn,
                                         decode_step=None, 
                                         attn_mask=m[[i]])
                
                # oo attn
                oo_k = obj_k[:, :, -1].flatten(2, 3) # B(n_head)(K(GG+5))D
                oo_v = obj_v[:, :, -1].flatten(2, 3)

                m = obj_oo_mask[-1]
                i = k_i * (self.bbox_n + 5) + self.bbox_n
                if shape_i == 1:
                    i += 1 + decode_idx[2]
                p_oo = self.perform_attn(q, oo_k, oo_v, self.full_attn,
                                         decode_step=None, 
                                         attn_mask=m[[i]])
                q = (p_ot + p_oo) / 2
            else: # base
                # bb attn
                p_bb = self.perform_attn(q, self.cache['k_base'], self.cache['v_base'],
                                         self.base_attn, 
                                         decode_step=flatten_idx(decode_idx, self.base_shape),
                                         flatten=True,
                                         attn_mask=base_attn_mask)

                # bo attn
                obj_k = obj_k[:, :, -1].flatten(2, 3) # B(n_head)(K(GG+5))D
                obj_v = obj_v[:, :, -1].flatten(2, 3)
                p_bo = self.perform_attn(q, obj_k, obj_v, self.full_attn, None)

                q = (p_bb + p_bo) / 2
            
            if has_cond:
                q = flatten((*cond, q))[0]
            return q 

        # B(n_head)T_1K5D, B(n_head)T_2HWD, B(n_head)T_1KGGD
        tensor_stats = aux_info['tensor_stats']
        bboxes_q, base_q, glimpses_q = expand(q, tensor_stats, has_head=True)
        bboxes_k, base_k, glimpses_k = expand(k, tensor_stats, has_head=True)
        bboxes_v, base_v, glimpses_v = expand(v, tensor_stats, has_head=True)

        obj_q = self._combine_obj(glimpses_q, bboxes_q) # B(n_head)T_1K(GG+5)D
        obj_k = self._combine_obj(glimpses_k, bboxes_k)
        obj_v = self._combine_obj(glimpses_v, bboxes_v)
        T1, K, GGp5 = obj_q.shape[2:5]
        T2 = base_q.shape[2]

        # base-base
        # BT_2HWD
        p_bb = self.perform_attn(base_q, base_k, base_v, self.base_attn,
                                 decode_step, flatten=True, attn_mask=base_attn_mask)
        
        # base-obj
        p_bo = self.perform_attn(base_q.flatten(start_dim=3, end_dim=4),
                                 obj_k[:, :, -T2:].flatten(start_dim=3, end_dim=4),
                                 obj_v[:, :, -T2:].flatten(start_dim=3, end_dim=4), 
                                 self.full_attn, None) # BT_2(HW)D
        p_bo = view_range(p_bo, 2, 3, (base_q.shape[3], base_q.shape[4])) # BT_2HWD

        p_base = (p_bb + p_bo) / 2 # BT_2HWD

        # obj-obj over time
        ot_q = obj_q.movedim(2, 3).flatten(start_dim=3, end_dim=4) # B(n_head)K(T_1(GG+5))D
        ot_k = obj_k.movedim(2, 3).flatten(start_dim=3, end_dim=4)
        ot_v = obj_v.movedim(2, 3).flatten(start_dim=3, end_dim=4)

        # BK(T(GG+5))D
        p_ot = self.perform_attn(ot_q, ot_k, ot_v, self.full_attn,
                                 decode_step=None, attn_mask=obj_ot_mask)
        p_ot = view_range(p_ot, 2, 3, (T1, GGp5)) # BKT_1(GG+5)D
        p_ot = p_ot.movedim(2, 1) # BT_1K(GG+5)D

        # obj-obj over obj
        ok_q = obj_q.flatten(start_dim=3, end_dim=4) # B(n_head)T_1(K(GG+5))D
        ok_k = obj_k.flatten(start_dim=3, end_dim=4)
        ok_v = obj_v.flatten(start_dim=3, end_dim=4)

        # BT(K(GG+5))D
        p_ok = self.perform_attn(ok_q, ok_k, ok_v, self.full_attn,
                                 decode_step=None, attn_mask=obj_oo_mask)
        p_ok = view_range(p_ok, 2, 3, (K, GGp5)) # BT_1K(GG+5)D
        
        p_oo = (p_ot + p_ok) / 2 # BT_1K(GG+5)D
        
        
        p_oo = p_oo.flatten(start_dim=2, end_dim=3) # BT_1(K(GG+5))D
        p_oo = view_range(p_oo, 2, 3, (K, GGp5)) # BT_1K(GG+5)D
        p_glimpse, p_bbox = torch.split(p_oo, [GGp5 - 5, 5], dim=3) # BT_1K(GG)D, BT_1K5D
        p_glimpse = view_range(p_glimpse, 3, 4, self.bbox_shape) # BT_1KGGD

        x = flatten((p_bbox, p_base, p_glimpse))[0]
        return x
        

class GeLU2(nn.Module):
    def forward(self, x):
        return (1.702 * x).sigmoid() * x


class LayerNorm(nn.Module):
    def __init__(self, embd_dim, class_cond_dim=None):
        super().__init__()
        self.conditional = class_cond_dim is not None

        if self.conditional:
            self.w = nn.Linear(class_cond_dim, embd_dim, bias=False)
            nn.init.constant_(self.w.weight.data, 1. / np.sqrt(class_cond_dim))
            self.wb = nn.Linear(class_cond_dim, embd_dim, bias=False)
        else:
            self.g = nn.Parameter(torch.ones(embd_dim, dtype=torch.float32), requires_grad=True)
            self.b = nn.Parameter(torch.zeros(embd_dim, dtype=torch.float32), requires_grad=True)

    def forward(self, x, cond=None):
        if self.conditional:  # (b, cond_dim)
            g = 1 + self.w(cond['class_cond']).view(x.shape[0], *(1,)*(len(x.shape)-2), x.shape[-1]) # (b, ..., embd_dim)
            b = self.wb(cond['class_cond']).view(x.shape[0], *(1,)*(len(x.shape)-2), x.shape[-1])
        else:
            g = self.g  # (embd_dim,)
            b = self.b

        x_float = x.float()

        mu = x_float.mean(dim=-1, keepdims=True)
        s = (x_float - mu).square().mean(dim=-1, keepdims=True)
        x_float = (x_float - mu) * (1e-5 + s.rsqrt())  # (b, ..., embd_dim)
        x_float = x_float * g + b

        x = x_float.type_as(x)
        return x


class FullAttention(nn.Module):
    def __init__(self, shape, causal, attn_dropout):
        super().__init__()
        self.causal = causal
        self.attn_dropout = attn_dropout

    def forward(self, q, k, v, decode_step, attn_mask):
        mask = attn_mask
        if decode_step is not None and mask is not None:
            if len(mask.shape) == 4:
                mask = mask[:, :, [decode_step]]
            else:
                mask = mask[[decode_step]]

        out = scaled_dot_product_attention(q, k, v, mask=mask,
                                           attn_dropout=self.attn_dropout,
                                           training=self.training)
        return out


def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0., training=True):
    # Performs scaled dot-product attention over the second to last dimension dn

    # (b, n_head, d1, ..., dn, d)
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / np.sqrt(q.shape[-1])
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float('-inf'))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn) # b x n_head x d1 x ... x dn x d
    attn = F.dropout(attn, p=attn_dropout, training=training)

    a = torch.matmul(attn, v) # b x n_head x d1 x ... x dn x d

    return a