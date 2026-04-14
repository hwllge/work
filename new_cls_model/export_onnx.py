"""
export_onnx.py  –  rps_mlp.pt → rps_mlp.onnx 변환

라즈베리파이(PyTorch 설치된 곳)에서 실행:
    python export_onnx.py
    python export_onnx.py --model rps_mlp.pt --out rps_mlp.onnx --hidden 64 32
"""

import argparse
import torch
import torch.nn as nn

NUM_FEATURES = 19


class RPSMLP(nn.Module):
    def __init__(self, input_dim=NUM_FEATURES, hidden_dims=(64, 32), num_classes=3, dropout=0.0):
        super().__init__()
        layers, dim = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="rps_mlp.pt")
    parser.add_argument("--out",    default="rps_mlp.onnx")
    parser.add_argument("--hidden", nargs="+", type=int, default=[64, 32])
    args = parser.parse_args()

    model = RPSMLP(hidden_dims=tuple(args.hidden), dropout=0.0)
    model.load_state_dict(torch.load(args.model, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.zeros(1, NUM_FEATURES)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"저장 완료: {args.out}")


if __name__ == "__main__":
    main()
