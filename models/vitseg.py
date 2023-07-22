import math
from posixpath import basename, dirname, join
# import clip
from clip.model import convert_weights
import torch
import json
from torch import nn
from torch.nn import functional as nnf
from torch.nn.modules import activation
from torch.nn.modules.activation import ReLU
from torchvision import transforms

normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

from torchvision.models import ResNet


def process_prompts(conditional, prompt_list, conditional_map):
    # DEPRECATED
            
    # randomly sample a synonym
    words = [conditional_map[int(i)] for i in conditional]
    words = [syns[torch.multinomial(torch.ones(len(syns)), 1, replacement=True).item()] for syns in words]
    words = [w.replace('_', ' ') for w in words]

    if prompt_list is not None:
        prompt_indices = torch.multinomial(torch.ones(len(prompt_list)), len(words), replacement=True)
        prompts = [prompt_list[i] for i in prompt_indices]
    else:
        prompts = ['a photo of {}'] * (len(words))

    return [promt.format(w) for promt, w in zip(prompts, words)]


class VITDenseBase(nn.Module):
    
    def rescaled_pos_emb(self, new_size):
        assert len(new_size) == 2

        a = self.model.positional_embedding[1:].T.view(1, 768, *self.token_shape)
        b = nnf.interpolate(a, new_size, mode='bicubic', align_corners=False).squeeze(0).view(768, new_size[0]*new_size[1]).T
        return torch.cat([self.model.positional_embedding[:1], b])

    def visual_forward(self, x_inp, extract_layers=(), skip=False, mask=None):
        
        with torch.no_grad():

            x_inp = nnf.interpolate(x_inp, (384, 384))

            x = self.model.patch_embed(x_inp)
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            if self.model.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = self.model.pos_drop(x + self.model.pos_embed)

            activations = []
            for i, block in enumerate(self.model.blocks):
                x = block(x)

                if i in extract_layers:
                    # permute to be compatible with CLIP
                    activations += [x.permute(1,0,2)]                

            x = self.model.norm(x)
            x = self.model.head(self.model.pre_logits(x[:, 0]))

            # again for CLIP compatibility
            # x = x.permute(1, 0, 2)

        return x, activations, None

    def sample_prompts(self, words, prompt_list=None):

        prompt_list = prompt_list if prompt_list is not None else self.prompt_list

        prompt_indices = torch.multinomial(torch.ones(len(prompt_list)), len(words), replacement=True)
        prompts = [prompt_list[i] for i in prompt_indices]
        return [promt.format(w) for promt, w in zip(prompts, words)]

    def get_cond_vec(self, conditional, batch_size):
        # compute conditional from a single string
        if conditional is not None and type(conditional) == str:
            cond = self.compute_conditional(conditional)
            cond = cond.repeat(batch_size, 1)

        # compute conditional from string list/tuple
        elif conditional is not None and type(conditional) in {list, tuple} and type(conditional[0]) == str:
            assert len(conditional) == batch_size
            cond = self.compute_conditional(conditional)

        # use conditional directly
        elif conditional is not None and type(conditional) == torch.Tensor and conditional.ndim == 2:
            cond = conditional

        # compute conditional from image
        elif conditional is not None and type(conditional) == torch.Tensor:
            with torch.no_grad():
                cond, _, _ = self.visual_forward(conditional)
        else:
            raise ValueError('invalid conditional')
        return cond   

    def compute_conditional(self, conditional):
        import clip

        dev = next(self.parameters()).device

        if type(conditional) in {list, tuple}:
            text_tokens = clip.tokenize(conditional).to(dev)
            cond = self.clip_model.encode_text(text_tokens)
        else:
            if conditional in self.precomputed_prompts:
                cond = self.precomputed_prompts[conditional].float().to(dev)
            else:
                text_tokens = clip.tokenize([conditional]).to(dev)
                cond = self.clip_model.encode_text(text_tokens)[0]
        
        return cond


class VITDensePredT(VITDenseBase):

    def __init__(self, extract_layers=(3, 6, 9), cond_layer=0, reduce_dim=128, n_heads=4, prompt='fixed', 
                 depth=3, extra_blocks=0, reduce_cond=None, fix_shift=False,
                 learn_trans_conv_only=False, refine=None, limit_to_clip_only=False, upsample=False, 
                 add_calibration=False, process_cond=None, not_pretrained=False):
        super().__init__()
        # device = 'cpu'

        self.extract_layers = extract_layers
        self.cond_layer = cond_layer
        self.limit_to_clip_only = limit_to_clip_only
        self.process_cond = None
        
        if add_calibration:
            self.calibration_conds = 1

        self.upsample_proj = nn.Conv2d(reduce_dim, 1, kernel_size=1) if upsample else None

        self.add_activation1 = True

        import timm 
        self.model = timm.create_model('vit_base_patch16_384', pretrained=True)
        self.model.head = nn.Linear(768, 512 if reduce_cond is None else reduce_cond)

        for p in self.model.parameters():
            p.requires_grad_(False)

        import clip
        self.clip_model, _ = clip.load('ViT-B/16', device='cpu', jit=False)
        # del self.clip_model.visual
        
        
        self.token_shape = (14, 14)

        # conditional
        if reduce_cond is not None:
            self.reduce_cond = nn.Linear(512, reduce_cond)
            for p in self.reduce_cond.parameters():
                p.requires_grad_(False)
        else:
            self.reduce_cond = None

        # self.film = AVAILABLE_BLOCKS['film'](512, 128)
        self.film_mul = nn.Linear(512 if reduce_cond is None else reduce_cond, reduce_dim)
        self.film_add = nn.Linear(512 if reduce_cond is None else reduce_cond, reduce_dim)
        
        # DEPRECATED
        # self.conditional_map = {c['id']: c['synonyms'] for c in json.load(open(cond_map))}
        
        assert len(self.extract_layers) == depth

        self.reduces = nn.ModuleList([nn.Linear(768, reduce_dim) for _ in range(depth)])
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads) for _ in range(len(self.extract_layers))])
        self.extra_blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=reduce_dim, nhead=n_heads) for _ in range(extra_blocks)])

        trans_conv_ks = (16, 16)
        self.trans_conv = nn.ConvTranspose2d(reduce_dim, 1, trans_conv_ks, stride=trans_conv_ks)

        # refinement and trans conv

        if learn_trans_conv_only:
            for p in self.parameters():
                p.requires_grad_(False)
            
            for p in self.trans_conv.parameters():
                p.requires_grad_(True)

        if prompt == 'fixed':
            self.prompt_list = ['a photo of a {}.']
        elif prompt == 'shuffle':
            self.prompt_list = ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.']
        elif prompt == 'shuffle+':
            self.prompt_list = ['a photo of a {}.', 'a photograph of a {}.', 'an image of a {}.', '{}.',
                                'a cropped photo of a {}.', 'a good photo of a {}.', 'a photo of one {}.',
                                'a bad photo of a {}.', 'a photo of the {}.']
        elif prompt == 'shuffle_clip':
            from models.clip_prompts import imagenet_templates
            self.prompt_list = imagenet_templates

        if process_cond is not None:
            if process_cond == 'clamp' or process_cond[0] == 'clamp':

                val = process_cond[1] if type(process_cond) in {list, tuple} else 0.2

                def clamp_vec(x):
                    return torch.clamp(x, -val, val)

                self.process_cond = clamp_vec

            elif process_cond.endswith('.pth'):
                
                shift = torch.load(process_cond)
                def add_shift(x):
                    return x + shift.to(x.device)

                self.process_cond = add_shift

        import pickle
        precomp = pickle.load(open('precomputed_prompt_vectors.pickle', 'rb'))
        self.precomputed_prompts = {k: torch.from_numpy(v) for k, v in precomp.items()}


    def forward(self, inp_image, conditional=None, return_features=False, mask=None):

        assert type(return_features) == bool

        # inp_image = inp_image.to(self.model.positional_embedding.device)

        if mask is not None:
            raise ValueError('mask not supported')

        # x_inp = normalize(inp_image)
        x_inp = inp_image

        bs, dev = inp_image.shape[0], x_inp.device

        inp_image_size = inp_image.shape[2:]

        cond = self.get_cond_vec(conditional, bs)

        visual_q, activations, _ = self.visual_forward(x_inp, extract_layers=[0] + list(self.extract_layers))

        activation1 = activations[0]
        activations = activations[1:]

        a = None
        for i, (activation, block, reduce) in enumerate(zip(activations[::-1], self.blocks, self.reduces)):
            
            if a is not None:
                a = reduce(activation) + a
            else:
                a = reduce(activation)

            if i == self.cond_layer:
                if self.reduce_cond is not None:
                    cond = self.reduce_cond(cond)
                
                a = self.film_mul(cond) * a + self.film_add(cond)

            a = block(a)

        for block in self.extra_blocks:
            a = a + block(a)

        a = a[1:].permute(1, 2, 0) # rm cls token and -> BS, Feats, Tokens

        size = int(math.sqrt(a.shape[2]))

        a = a.view(bs, a.shape[1], size, size)

        if self.trans_conv is not None:
            a = self.trans_conv(a)

        if self.upsample_proj is not None:
            a = self.upsample_proj(a)
            a = nnf.interpolate(a, x_inp.shape[2:], mode='bilinear')

        a = nnf.interpolate(a, inp_image_size)

        if return_features:
            return a, visual_q, cond, [activation1] + activations
        else:
            return a,
