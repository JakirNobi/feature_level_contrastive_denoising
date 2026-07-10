import torch
from ultralytics import YOLO

model = YOLO('yolo26s.pt')
det = model.model
assert det is not None, "DetectionModel not found"

layers = det.model  # nn.Sequential
assert isinstance(layers, torch.nn.Sequential)

# Dummy input
x = torch.randn(1, 3, 640, 640)

# Dictionary to store outputs by layer index
outputs = {}

def hook_fn(module, input, output, layer_idx):
    outputs[layer_idx] = output

handles = []
for i, layer in enumerate(layers):
    h = layer.register_forward_hook(lambda mod, inp, out, idx=i: outputs.__setitem__(idx, out))
    handles.append(h)

# Single forward pass
with torch.no_grad():
    det(x)

# Remove hooks
for h in handles:
    h.remove()

# Now find the target feature maps
print("\nPossible P3/P4/P5 candidates:")
for idx, out in outputs.items():
    if isinstance(out, (list, tuple)):
        for j, o in enumerate(out):
            if isinstance(o, torch.Tensor) and o.dim() == 4 and o.shape[2] == o.shape[3]:
                c, h = o.shape[1], o.shape[2]
                if h in [80, 40, 20]:
                    print(f"Layer {idx}[{j}]: shape {o.shape} -> P{c//128 if h==80 else c//64 if h==40 else c//256}")
    elif isinstance(out, torch.Tensor) and out.dim() == 4 and out.shape[2] == out.shape[3]:
        _, c, h, _ = out.shape
        if h == 80 and c == 128:
            print(f"Layer {idx}: shape {out.shape} -> ** P3 **")
        elif h == 40 and c == 256:
            print(f"Layer {idx}: shape {out.shape} -> ** P4 **")
        elif h == 20 and c == 512:
            print(f"Layer {idx}: shape {out.shape} -> ** P5 **")