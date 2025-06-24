# export_detr_onnx.py
#
# pip3 install transformers timm
# python3 export_detr_onnx.py
# trtexec --onnx=detr.onnx --saveEngine=detr.engine --fp16
#

from transformers import DetrForObjectDetection
import torch

model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.eval()

dummy_input = torch.randn(1, 3, 720, 1280)

torch.onnx.export(
    model,
    dummy_input,
    "detr.onnx",
    input_names=["pixel_values"],
    output_names=["logits", "boxes"],
    opset_version=11,
    dynamic_axes={"pixel_values": {0: "batch_size"},
                  "logits": {0: "batch_size"},
                  "boxes": {0: "batch_size"}}
)
print("Exported to detr.onnx")
