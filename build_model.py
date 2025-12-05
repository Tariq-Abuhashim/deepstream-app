'''

Assuming image: 720 x 1280
Network dimensions becomes: 800 x 1422
On CPU:

python3 build_model.py \
    --checkpoint weights/window_res101_cp1399.pth \
    --num_classes 1 \
    --out build/detr.onnx \
    --w 1280 \
    --h 720 \
    --device cpu
    
If you want GPU

python3 build_model.py \
    --checkpoint weights/window_res101_cp1399.pth \
    --num_classes 1 \
    --out build/detr.onnx \
    --w 1280 \
    --h 720 \
    --device cuda
    
# Simplify ONNX model
python3 -m onnxsim build/detr.onnx build/detr_simplified.onnx

# Convert with TRT (with memory limits)
trtexec --onnx=build/detr_simplified.onnx \
        --saveEngine=build/detr.engine \
        --fp16 \
        --workspace=2048 \
        --shapes=input:1x3x720x1280 \
        --verbose=3 \
        --tacticSources=+CUDNN,-CUBLAS_LT
'''


import sys
import argparse
from pathlib import Path
import torch

# Append the path to the DETR repository
sys.path.append("detr")  # Replace with the path to your cloned DETR directory
from models import build_model

# Model setup
class Args:
    def __init__(self, num_classes=1, device="cuda"):
        self.device = device
        self.num_classes = num_classes

        # Required by build_model()
        self.backbone = "resnet101"        # must match training
        self.dilation = False
        self.position_embedding = "sine"
        self.lr_backbone = 1e-5 #Learning rate for the backbone network.

        # Transformer params (must match checkpoint)
        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.num_queries = 100
        self.activation = 'relu'
        self.pre_norm = False
        self.normalize_before = True

        # * Segmentation
        self.masks = False #Whether or not to use segmentation masks.

        # * Loss
        self.aux_loss = True #indicates whether or not to use auxiliary decoding losses (in addition to the usual decoding loss). These can help improve the training stability of the DETR model.
        
        # * Matcher
        self.set_cost_class = 1.0 #Class coefficient in the matching cost
        self.set_cost_bbox = 5.0 #L1 box coefficient in the matching cost
        self.set_cost_giou = 2.0 #GIoU box coefficient in the matching cost
        
        # * Loss coefficients
        self.mask_loss_coef = 1.0
        self.dice_loss_coef = 1.0
        self.bbox_loss_coef = 5.0 #Coefficient for the bounding box loss in the total loss function.
        self.giou_loss_coef = 2.0
        self.eos_coef = 0.1 #Relative classification weight of the no-object class")

        # Misc required placeholders
        self.dataset_file = "custom_dataset"
        
    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(description='Tracks windows in images and computes their normals.')
        parser.add_argument('--w', type=int, default=1273, help='Width of the input images') #vulcan 1274, vulcan-mono-slam 1273
        parser.add_argument('--h', type=int, default=800, help='Height of the input images')
        parser.add_argument('--num_classes', type=int, default=1, help='Number of classes')
        parser.add_argument('--checkpoint', required=True, type=str, default="weights/window_res101_cp1399.pth", help='Path to the trained model checkpoint')
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--out", default="detr.onnx")
        return parser.parse_args()

def load_detr(args):

    # Build empty model
    model, _, _ = build_model(args)
    model.to(args.device)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model.eval()
    return model
    
def export_onnx(model, out, h, w, device):
    dummy = torch.randn(1, 3, h, w).to(device)

    torch.onnx.export(
        model,
        dummy,
        out,
        opset_version=11,
        input_names=["input"],
        output_names=["outputs"], # the model's output names
        dynamic_axes={'input': {0: 'batch_size'}, 'outputs': {0: 'batch_size'}}
    )
    print(f"Exported ONNX model: {out}")

def main():
    inargs = Args.parse_args()

    args = Args()
    args.w = inargs.w
    args.h = inargs.h
    args.checkpoint = inargs.checkpoint
    args.num_classes = inargs.num_classes #over-writes number of classes using user input
    args.out = inargs.out
    args.device = inargs.device
    
    model = load_detr(args)
    export_onnx(model, args.out, args.h, args.w, args.device)

if __name__ == "__main__":
    main()
