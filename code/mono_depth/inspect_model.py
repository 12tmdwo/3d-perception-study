"""모델 해부: Depth Anything V2 Small의 부품·크기·매개변수 수를 직접 출력한다. python3 code/mono_depth/inspect_model.py"""
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from common import HERE, RELATIVE_MODEL, METRIC_MODEL


def count(m):   # 매개변수 수 (백만 단위)
    return sum(p.numel() for p in m.parameters()) / 1e6


for name in [RELATIVE_MODEL, METRIC_MODEL]:
    model = AutoModelForDepthEstimation.from_pretrained(name).eval()
    c, b = model.config, model.config.backbone_config
    print(f"\n== {name}")
    print(f"출력 방식: {c.depth_estimation_type}, max_depth={c.max_depth}, 마지막 활성함수: {type(model.head.activation2).__name__}")
    print(f"뼈대: {b.model_type} 차원={b.hidden_size} 층={b.num_hidden_layers} 헤드={b.num_attention_heads} 조각={b.patch_size} 꺼내는 층={b.out_indices}")
    print(f"목: 채널={c.neck_hidden_sizes} 배율={c.reassemble_factors} 융합채널={c.fusion_hidden_size}")
    print(f"매개변수(백만): 전체 {count(model):.2f} = 뼈대 {count(model.backbone):.2f} + 목 {count(model.neck):.2f} + 머리 {count(model.head):.2f}")

shapes = []
model.neck.register_forward_hook(lambda m, i, o: shapes.extend(tuple(t.shape[1:]) for t in o))
proc = AutoImageProcessor.from_pretrained(RELATIVE_MODEL)
rgb = cv2.cvtColor(cv2.imread(str(HERE / "test.jpg")), cv2.COLOR_BGR2RGB)
x = proc(images=Image.fromarray(rgb), return_tensors="pt")["pixel_values"]
with torch.no_grad():
    y = model(pixel_values=x).predicted_depth
h, w = x.shape[2] // 14, x.shape[3] // 14
print(f"\n입력 {rgb.shape[1]}x{rgb.shape[0]} → 모델 입력 {tuple(x.shape[1:])} → 조각 {h}x{w}={h * w}개 → 원본 출력 {tuple(y.shape[1:])}")
print("목이 만든 특징 지도 (채널, 세로, 가로):", shapes)
print("\n[뼈대의 한 층]")
print(model.backbone.encoder.layer[0])
print("\n[머리]")
print(model.head)
