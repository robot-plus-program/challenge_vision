# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
if "CustomRCNN" not in META_ARCH_REGISTRY:
    from .modeling.meta_arch import custom_rcnn
else:
    print("CustomRCNN is already registered in META_ARCH_REGISTRY")
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY

if "DeticCascadeROIHeads" not in ROI_HEADS_REGISTRY:
    from .modeling.roi_heads import detic_roi_heads
else:
    print("DeticCascadeROIHeads is already registered in META_ARCH_REGISTRY")


if "CustomRes5ROIHeads" not in ROI_HEADS_REGISTRY:
    from .modeling.roi_heads import res5_roi_heads
else:
    print("CustomRes5ROIHeads is already registered in META_ARCH_REGISTRY")

from detectron2.modeling.backbone.build import BACKBONE_REGISTRY

if "build_swintransformer_backbone" not in BACKBONE_REGISTRY:
    from .modeling.backbone import swintransformer
else:
    print("build_swintransformer_backbone is already registered in META_ARCH_REGISTRY")



if "build_timm_backbone" not in BACKBONE_REGISTRY:
    from .modeling.backbone import timm
else:
    print("build_timm_backbone is already registered in META_ARCH_REGISTRY")


try:
    from .data.datasets import lvis_v1
    from .data.datasets import imagenet
    from .data.datasets import cc
    from .data.datasets import objects365
    from .data.datasets import oid
    from .data.datasets import coco_zeroshot
except:
    pass
try:
    from .modeling.meta_arch import d2_deformable_detr
except:
    pass