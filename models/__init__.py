from .cnn.L2HNet import L2HNetSeg
from .cnn.SkipFCN import Skip_FCN
from .cnn.VGG import Dof_VGG
from .doft.DofT import DofTV1
from .hf_model.DPT import DPT
from .hf_model.Mask2Former import Mask2Former
from .hf_model.SegFormer import SegFormer, DofSegFormer, LORASegFormer
from .proto.HRNet_Proto import HRNet_W48_Proto
from .unet.MobileUNet import MobileUNet


def build_models(model_name, in_channels, num_classes, image_size, **kwargs):
    if model_name == 'Mask2Former':
        model_id = kwargs.pop('model_id')
        return Mask2Former(model_id=Mask2Former.get_model_ids()[model_id],
                            in_chans=in_channels, num_classes=num_classes)

    if model_name == 'HRNetProto':
        return HRNet_W48_Proto(num_channels=in_channels, num_classes=num_classes)

    if model_name == 'DofSegFormer':
        model_id = kwargs.pop('model_id')
        return DofSegFormer(model_id=DofSegFormer.get_model_ids()[model_id],
                            in_chans=in_channels, num_classes=num_classes)

    if model_name == 'LORASegFormer':
        return LORASegFormer(model_id=SegFormer.get_model_ids()[-1],
                         in_chans=in_channels, num_classes=num_classes)

    if model_name == 'SegFormer':
        return SegFormer(model_id=SegFormer.get_model_ids()[-1],
                         in_chans=in_channels, num_classes=num_classes)

    if model_name == 'DPT':
        return DPT(model_id=DPT.get_model_ids()[1],
                   in_chans=in_channels, num_classes=num_classes)

    if model_name == 'Dof_VGG':
        return Dof_VGG(in_chans=in_channels, num_classes=num_classes, img_size=image_size)

    if model_name == 'DofT':
        return DofTV1(in_chans=in_channels, num_classes=num_classes, img_size=image_size)

    if model_name == 'SkipFCN':
        return Skip_FCN(num_input_channels=in_channels, num_output_classes=num_classes)

    if model_name == 'L2HNet':
        return L2HNetSeg(in_chans=in_channels, num_classes=num_classes)

    if model_name == 'MobileUNet':
        return MobileUNet(in_chans=in_channels, num_classes=num_classes)

    pass
