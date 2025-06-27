'''
export_detr_onnx.py

# Build the onnx model
pip3 install transformers timm
python3 export_detr_onnx.py

# Simplify ONNX model
python3 -m onnxsim detr.onnx detr_simplified.onnx

# Convert with TRT (with memory limits)
trtexec --onnx=detr_simplified.onnx \
        --saveEngine=detr.engine \
        --fp16 \
        --workspace=2048 \
        --shapes=pixel_values:1x3x800x1273 \
        --verbose=3 \
        --tacticSources=+CUDNN,-CUBLAS_LT
'''

from transformers import DetrForObjectDetection
import torch

# Load model and move to CUDA
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").eval().cuda()  # <- Critical change

# Create input tensor on CUDA
dummy_input = torch.randn(1, 3, 720, 1280, device='cuda')  # Must match model device

# Export with device consistency
torch.onnx.export(
    model,
    dummy_input,
    "detr.onnx",
    input_names=["pixel_values"],
    output_names=["logits", "boxes"],
    opset_version=13,
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "logits": {0: "batch_size"},
        "boxes": {0: "batch_size"}
    },
    do_constant_folding=True,
    training=torch.onnx.TrainingMode.EVAL
)

print("Successfully exported to detr.onnx")
