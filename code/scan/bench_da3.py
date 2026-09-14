"""0a 측정: DA3-SMALL(장수·해상도·정밀도), DA3METRIC-LARGE, YOLO26n-seg의 속도와 그래픽카드 메모리.

실행: cd code/scan && ~/venvs/da3/bin/python bench_da3.py 2>&1 | tee bench_da3.txt
"""
import argparse
import functools
import os
import time
from pathlib import Path

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")  # DA3 INFO 로그가 표를 가려서

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
YOLO_PATH = HERE.parents[1] / "yolo26n-seg.pt"
EXPECT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
_AUTOCAST, _BF16 = torch.autocast, torch.cuda.is_bf16_supported


def set_precision(p):
    # ponytail: DA3 api.forward가 autocast를 코드에 박아둬서 torch 쪽을 바꿔치기함. 정밀도가 정해지면 DA3에 옵션을 넣는 게 정석
    torch.autocast, torch.cuda.is_bf16_supported = _AUTOCAST, _BF16
    if p == "fp16":
        torch.cuda.is_bf16_supported = lambda *a, **k: False
    elif p == "fp32":
        torch.autocast = functools.partial(_AUTOCAST, enabled=False)


def probe_dtype():
    """DA3 api.forward와 같은 두 줄로 실제 계산 dtype 확인."""
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.autocast(device_type="cuda", dtype=dt):
        x = torch.ones(2, 2, device="cuda")
        return torch.nn.functional.linear(x, x).dtype


def load_rgb(paths, n):
    """640x480 RGB n장. 사진이 모자라면 옆으로 조금씩 밀어서 반복(완전히 같은 사진 방지)."""
    base = []
    for p in paths:
        bgr = cv2.imread(str(p))
        assert bgr is not None, f"사진을 못 읽음: {p}"
        base.append(cv2.cvtColor(cv2.resize(bgr, (640, 480)), cv2.COLOR_BGR2RGB))
    return [np.roll(base[i % len(base)], 16 * i, axis=1) for i in range(n)]


def timed(fn, runs, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t))
    return np.mean(ts), np.std(ts), torch.cuda.max_memory_allocated() / 2**20, out


def row(name, prec, res, n, ms, sd, mib, extra=""):
    print(f"{name:<16}{prec:<6}{res:>5}{n:>4}장 | {ms:7.0f} ±{sd:5.0f} ms | {ms / n:6.0f} ms/장 | "
          f"최대 {mib:6.0f} MiB  {extra}", flush=True)


def rot_deg(E):
    return np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1)))


def main():
    ints = lambda s: [int(x) for x in s.split(",")]
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=[HERE.parent / "mono_depth" / "test.jpg"])
    ap.add_argument("--views", type=ints, default=[1, 2, 3, 4, 8, 16, 32, 64])
    ap.add_argument("--res", type=ints, default=[504, 392, 280])
    ap.add_argument("--prec", type=lambda s: s.split(","), default=["fp32", "bf16", "fp16"])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    for p in args.prec:  # 자체 검사: 바꿔치기가 실제로 먹히는지
        set_precision(p)
        assert probe_dtype() == EXPECT[p], (p, probe_dtype())
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)} | 정밀도 바꿔치기 검사 통과\n")

    from depth_anything_3.api import DepthAnything3
    from ultralytics import YOLO

    model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to("cuda").eval()
    seen = {}  # 모델 안 첫 Linear의 실제 출력 dtype
    lin = next(m for m in model.modules() if isinstance(m, torch.nn.Linear))
    lin.register_forward_hook(lambda m, i, o: seen.update(dtype=str(o.dtype).replace("torch.", "")))
    two = load_rgb(args.images, 2)
    ref_depth, checked = None, False

    for p in args.prec:
        set_precision(p)
        print(f"--- DA3-SMALL {p}: 장수 늘리기 (res {args.res[0]}) ---")
        for n in args.views:
            imgs = load_rgb(args.images, n)
            r, w = (args.runs, args.warmup) if n < 8 else (2, 1)
            try:
                ms, sd, mib, pred = timed(
                    lambda: model.inference(imgs, process_res=args.res[0], ref_view_strategy="first"), r, w)
            except RuntimeError as e:  # 메모리 부족 포함
                print(f"{n}장에서 중단: {str(e).splitlines()[0][:120]}")
                torch.cuda.empty_cache()
                break
            extra = f"dtype={seen['dtype']}"
            if n == 2:
                d = pred.depth
                extra += f" 유한값={np.isfinite(d).all()}"
                if p == "fp32" and ref_depth is None:
                    ref_depth = d
                elif ref_depth is not None:
                    extra += f" fp32대비_깊이차={np.median(np.abs(d - ref_depth) / ref_depth):.3%}"
                if not checked:
                    checked = True
                    W = d.shape[2]
                    print(f"  모양 검사: depth {d.shape}, processed {pred.processed_images.shape}, "
                          f"크기일치={d.shape[1:] == pred.processed_images.shape[1:3]}, "
                          f"cx/W={pred.intrinsics[0, 0, 2] / W:.2f}, 기준사진 회전={rot_deg(pred.extrinsics[0]):.3f}°")
            row("DA3-SMALL", p, args.res[0], n, ms, sd, mib, extra)

        for res in args.res[1:]:
            ms, sd, mib, _ = timed(
                lambda: model.inference(two, process_res=res, ref_view_strategy="first"), args.runs, args.warmup)
            row("DA3-SMALL", p, res, 2, ms, sd, mib, f"dtype={seen['dtype']}")
        ms, sd, mib, _ = timed(lambda: model.inference(
            two, process_res=args.res[0], ref_view_strategy="first", use_ray_pose=True), args.runs, args.warmup)
        row("DA3-SMALL ray", p, args.res[0], 2, ms, sd, mib, f"dtype={seen['dtype']}")
        print()

    set_precision("bf16")  # DA3 기본 상태로
    print("--- YOLO26n-seg (imgsz 640, 메모리 칸은 DA3-SMALL 포함) ---")
    frame = cv2.cvtColor(two[0], cv2.COLOR_RGB2BGR)
    for half in (False, True):
        yolo = YOLO(str(YOLO_PATH))  # half 설정은 첫 predict에서 굳어서 매번 새로 불러옴
        ms, sd, mib, out = timed(lambda: yolo.predict(
            frame, imgsz=640, retina_masks=True, half=half, device=0, verbose=False), args.runs, args.warmup)
        row("YOLO26n-seg", "fp16" if half else "fp32", 640, 1, ms, sd, mib,
            f"클래스 {len(yolo.names)}종, 검출 {len(out[0].boxes)}개")
    torch.cuda.empty_cache()  # 앞 측정(ray 등)에서 잡아둔 캐시가 섞이지 않게
    free, total = torch.cuda.mem_get_info()
    print(f"DA3-SMALL + YOLO 함께: torch 할당 {torch.cuda.memory_allocated() / 2**20:.0f} MiB, "
          f"그래픽카드 전체 사용 {(total - free) / 2**20:.0f} / {total / 2**20:.0f} MiB\n")
    del model, yolo
    torch.cuda.empty_cache()

    print("--- DA3METRIC-LARGE 1장 ---")
    metric = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda").eval()
    for p in args.prec:
        set_precision(p)
        try:
            ms, sd, mib, pred = timed(lambda: metric.inference(two[:1], process_res=args.res[0]), args.runs, args.warmup)
        except RuntimeError as e:
            print(f"{p} 중단: {str(e).splitlines()[0][:120]}")
            torch.cuda.empty_cache()
            continue
        row("DA3METRIC-LARGE", p, args.res[0], 1, ms, sd, mib,
            f"출력 중앙값 {np.median(pred.depth):.3f} 유한값={np.isfinite(pred.depth).all()}")


if __name__ == "__main__":
    main()
