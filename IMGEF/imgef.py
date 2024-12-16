import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorflow.python.layers.core import dropout
from R2Gen.modules.visual_extractor import VisualExtractor
from R2Gen.modules.base_cmn import BaseCMN
from R2Gen.modules.integrlmodel import IntegratedModel


class imgef(nn.Module):
    def __init__(self, args, tokenizer, num_classes, forward_adj, backward_adj, feat_size=2048, embed_size=256,hidden_size=612,  vocab_size=10000):
        super(imgef, self).__init__()
        self.args = args
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.forward_adj = forward_adj
        self.backward_adj = backward_adj
        self.feat_size = feat_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = BaseCMN(args, tokenizer, )

        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        else:
            self.forward = self.forward_mimic_cxr

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def init_hidden(self, batch_size):
        device = torch.device('cuda')
        hidden_state = torch.zeros(self.args.num_layers, batch_size, self.hidden_size, device=device)
        cell_state = torch.zeros(self.args.num_layers, batch_size, self.hidden_size, device=device)
        return hidden_state, cell_state

    def forward(self, img_feats, enc_feats, captions,images, **kwargs):
        batch_size = img_feats.size(0)

       
        img_feats = self.positional_encoding(img_feats)
        context_feats, alpha = self.key_event_attention(enc_feats, img_feats)
        context_feats = context_feats.view(batch_size, self.feature_dim, 1, 1)
        visual_feats = self.visual_extractor(img_feats)
        visual_feats = self.class_attention(visual_feats)
        flattened_visual = visual_feats.view(batch_size, -1)
        self.hidden = (self.init_sent_h(flattened_visual), self.init_sent_c(flattened_visual))
        output_captions = self.captioning(visual_feats, captions)

        return output_captions, alpha

    def forward_iu_xray(self, images, captions=None, mode='train', update_opts={}):
        att_feats = []
        func_feats = []

        for i in range(2):
            att_feat, func_feat = self.visual_extractor(images[:, i])
            att_feats.append(att_feat)
            func_feats.append(func_feat)

        func_feat = torch.cat(func_feats, dim=1)
        forward_adj = self.normalized_forward_adj.repeat(8, 1, 1)
        backward_adj = self.normalized_backward_adj.repeat(8, 1, 1)
        global_feats = [feat.mean(dim=(2, 3)) for feat in att_feats]
        att_feats = [self.cls_atten(feat, self.num_classes) for feat in att_feats]

        for idx in range(2):
            att_feats[idx] = torch.cat((global_feats[idx].unsqueeze(1), att_feats[idx]), dim=1)
            att_feats[idx] = self.linear_trans_lyr_2(att_feats[idx].transpose(1, 2)).transpose(1, 2)
        att_feat_combined = torch.cat(att_feats, dim=1)
        att_feat_combined = self.linear_trans_lyr(att_feat_combined.transpose(1, 2)).transpose(1, 2)

        if mode == 'train':
            return self.encoder_decoder(func_feat, att_feat_combined, captions, mode='forward')
        elif mode == 'sample':
            return self.encoder_decoder(func_feat, att_feat_combined, mode='sample', update_opts=update_opts)
        else:
            raise ValueError("Invalid mode provided.")

    def forward_mimic_cxr(self, images, captions=None, mode='train', update_opts={}):
        att_feats, func_feat = self.visual_extractor(images)
        if mode == 'train':
            return self.encoder_decoder(func_feat, att_feats, captions, mode='forward')
        elif mode == 'sample':
            return self.encoder_decoder(func_feat, att_feats, mode='sample', update_opts=update_opts)
        else:
            raise ValueError("Invalid mode provided.")
