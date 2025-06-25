import warnings

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

# from lib.models.backbones.backbone_selector import BackboneSelector
# from lib.models.modules.contrast import momentum_update, l2_normalize, ProjectionHead
# from lib.models.modules.projection import ProjectionHead
# from lib.models.modules.sinkhorn import distributed_sinkhorn
# from lib.models.tools.module_helper import ModuleHelper
from .HRNetBackbone import HRNetBackbone
from .contrast import momentum_update, l2_normalize
from .sinkhorn import distributed_sinkhorn

warnings.filterwarnings("ignore")


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                # ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)


class HRNet_W48_Proto(nn.Module):
    """
    deep high-resolution representation learning for human pose estimation, CVPR2019

    "protoseg": {
      "gamma": 0.999,
      "loss_ppc_weight": 0.01,
      "loss_ppd_weight": 0.001,
      "num_prototype": 10,
      "pretrain_prototype": false,
      "use_rmi": false,
      "use_prototype": true,
      "update_prototype": true,
      "warmup_iters": 0
    }
    """

    def __init__(self, num_channels, num_classes):
        super(HRNet_W48_Proto, self).__init__()
        self.gamma = 0.999
        self.num_prototype = 10
        self.use_prototype = True
        self.update_prototype = True
        self.pretrain_prototype = False
        self.num_classes = num_classes

        config = r'F:\Projects2\SpiderNet\models\HRNet\config.yml'
        self.backbone = HRNetBackbone(in_channel=num_channels, num_classes=num_classes, cfg=config)

        in_channels = 720
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout2d(0.10)
        )

        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, in_channels),
                                       requires_grad=True)

        self.proj_head = ProjectionHead(in_channels, in_channels)
        self.feat_norm = nn.LayerNorm(in_channels)
        self.mask_norm = nn.LayerNorm(self.num_classes)

        trunc_normal_(self.prototypes, std=0.02)

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))

        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())

        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()

        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue

            q, indexs = distributed_sinkhorn(init_q)

            m_k = mask[gt_seg == k]

            c_k = _c[gt_seg == k, ...]

            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)

            m_q = q * m_k_tile  # n x self.num_prototype

            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])

            c_q = c_k * c_k_tile  # n x embedding_dim

            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim

            n = torch.sum(m_q, dim=0)

            if torch.sum(n) > 0 and self.update_prototype is True:
                f = F.normalize(f, p=2, dim=-1)

                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :],
                                            momentum=self.gamma, debug=False)
                protos[k, n != 0, :] = new_value

            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)

        self.prototypes = nn.Parameter(l2_normalize(protos),
                                       requires_grad=False)

        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

        return proto_logits, proto_target

    def forward(self, x_, gt_semantic_seg=None, pretrain_prototype=False):
        x = self.backbone(x_)
        _, _, h, w = x[0].size()

        feat1 = x[0]
        feat2 = F.interpolate(x[1], size=(h, w), mode="bilinear", align_corners=True)
        feat3 = F.interpolate(x[2], size=(h, w), mode="bilinear", align_corners=True)
        feat4 = F.interpolate(x[3], size=(h, w), mode="bilinear", align_corners=True)

        feats = torch.cat([feat1, feat2, feat3, feat4], 1)
        c = self.cls_head(feats)

        c = self.proj_head(c)
        _c = rearrange(c, 'b c h w -> (b h w) c')
        _c = self.feat_norm(_c)
        _c = l2_normalize(_c)

        self.prototypes.data.copy_(l2_normalize(self.prototypes))

        # n: h*w, k: num_class, m: num_prototype
        masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)

        out_seg = torch.amax(masks, dim=1)
        out_seg = self.mask_norm(out_seg)
        out_seg = rearrange(out_seg, "(b h w) k -> b k h w", b=feats.shape[0], h=feats.shape[2])
        out_seg = nn.functional.interpolate(out_seg, scale_factor=4, mode='bilinear', align_corners=False)

        if pretrain_prototype is False and self.use_prototype is True and gt_semantic_seg is not None:
            gt_seg = F.interpolate(gt_semantic_seg.float(), size=feats.size()[2:], mode='nearest').view(-1)
            contrast_logits, contrast_target = self.prototype_learning(_c, out_seg, gt_seg, masks)
            return {'seg': out_seg, 'logits': contrast_logits, 'target': contrast_target}

        return out_seg
