import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from nets.MobileNetV2_unet import MobileNetV2_unet

EXPERIMENT = 'train_mogiznet'
OUT_DIR = 'outputs/{}'.format(EXPERIMENT)


def save_to_mobile(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    torchscript_model = torch.jit.script(model)
    torchscript_model_optimized = optimize_for_mobile(torchscript_model)
    torch.jit.save(torchscript_model_optimized,
                   '{}/{}-best-mobile.pt'.format(OUT_DIR, 0))
    #torchscript_model_optimized._save_for_lite_interpreter('{}/{}-best.ptl'.format(OUT_DIR, 0))


model = MobileNetV2_unet(mode="eval", pre_trained=None)
#pretrained_model_hw = torch.load('saved_models/u2net_human_seg.pth', map_location='cpu')
pretrained_model_hw = torch.load('{}/0-best.pth'.format(OUT_DIR))
# print(pretrained_model_hw)
model.load_state_dict(pretrained_model_hw)
save_to_mobile(model)
