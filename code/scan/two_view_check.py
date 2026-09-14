"""0b: 고정 카메라 사진 + 이동 카메라 사진을 DA3-SMALL에 넣었을 때 두 카메라 위치 관계가 맞게 나오는지 확인.

두 카메라로 찍으며 평가: cd code/scan && ~/venvs/da3/bin/python two_view_check.py --cams   (SPACE = 찍고 평가, ESC = 끝)
저장된 사진 쌍 평가:     ~/venvs/da3/bin/python two_view_check.py --pair 고정.png 이동.png
자체 검사:              python3 two_view_check.py --selftest

보는 것: Open3D 창에서 빨강(고정)·파랑(이동) 점구름이 겹치는지 + 겹침 거리(overlap_med_mm).
두 사진 모두에 9x6 체커보드가 보이면 정답 비교(rot_err, dir_err, corner3d_mm …)도 출력한다.
K=calib(캘리브레이션 초점거리), K=da3(DA3 추정 초점거리) 두 줄씩 pairs/results.csv에 쌓인다.
"""
import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from scan_live import FIX_CALIB, FIX_CAM, MOVE_CALIB, MOVE_CAM, center, open_camera, rot_deg, scale_K, to_world
from common import FLIP, depth_to_points  # scan_live가 mono_depth 폴더를 sys.path에 넣어 둠

HERE = Path(__file__).resolve().parent
PAIRS = HERE / "pairs"
BOARD, SQUARE = (9, 6), 0.02   # 예전 캘리브레이션 보드: 안쪽 코너 9x6, 칸 2cm
OBJP = np.array([[x, y, 0] for y in range(BOARD[1]) for x in range(BOARD[0])], np.float32) * SQUARE
CONF_PCT = 50   # 임시 기준: 사진마다 DA3 신뢰도 상위 절반 픽셀만 사용
FIELDS = ["tag", "K", "E0_rot", "cam_angle", "baseline_m", "s", "overlap_frac", "spacing_mm", "overlap_med_mm",
          "overlap_pct", "gt_angle", "rot_err", "dir_err", "baseline_ratio", "metric_ratio", "corner3d_mm", "corner_gt_mm"]


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


def relative(R0, t0, R1, t1):
    """'월드→카메라' 변환 두 개에서 카메라0 좌표 → 카메라1 좌표 변환."""
    R = R1 @ R0.T
    return R, t1 - R @ t0


def unproject(uv, z, K):
    return np.column_stack(((uv[:, 0] - K[0, 2]) * z / K[0, 0], (uv[:, 1] - K[1, 2]) * z / K[1, 1], z))


def sample(img, c, W, H):
    """640x480 픽셀 좌표 c 위치의 값을 처리 해상도(W,H) 지도에서 가장 가까운 픽셀로 읽기."""
    x = np.clip(np.round(c[:, 0] * W / 640).astype(int), 0, W - 1)
    y = np.clip(np.round(c[:, 1] * H / 480).astype(int), 0, H - 1)
    return img[y, x]


def find_board(bgr):
    ok, c = cv2.findChessboardCornersSB(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), BOARD)
    return c.reshape(-1, 2) if ok else None


def solve_board(c, K):
    """보드→카메라 R, t와 재투영 오차(px).
    왜곡 계수는 안 씀: 기존 dist를 적용하면 이 C270의 가장자리 직선이 오히려 더 휘었다(2026-09-12 측정)."""
    _, rvec, t = cv2.solvePnP(OBJP, c, K, None)
    err = np.linalg.norm(cv2.projectPoints(OBJP, rvec, t, K, None)[0].reshape(-1, 2) - c, axis=1).mean()
    return cv2.Rodrigues(rvec)[0], t.ravel(), float(err)


def save_rows(rows):
    PAIRS.mkdir(exist_ok=True)
    path = PAIRS / "results.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS, restval="")
        if new:
            w.writeheader()
        w.writerows(rows)


def evaluate(bgrs, Ks, da3, metric, tag, show=True):
    """bgrs = [고정 사진, 이동 사진] (640x480 BGR), Ks = 두 카메라의 640x480 기준 K."""
    import open3d as o3d

    rgbs = [cv2.cvtColor(b, cv2.COLOR_BGR2RGB) for b in bgrs]
    p = da3.inference(rgbs, process_res=504, ref_view_strategy="first")
    raw = metric.inference(rgbs[:1], process_res=504).depth[0]
    H, W = p.depth.shape[1:]
    assert raw.shape == (H, W), (raw.shape, H, W)
    Rs, ts = p.extrinsics[:, :, :3], p.extrinsics[:, :, 3]
    Rr, tr = relative(Rs[0], ts[0], Rs[1], ts[1])
    C1 = center(Rr, tr)   # 이동 카메라 중심 (고정 카메라 좌표, DA3 단위)
    print(f"\n=== {tag} ===  기준사진 회전 {rot_deg(Rs[0]):.3f}° (0이어야 함) | DA3가 본 두 카메라 사이 각도 {rot_deg(Rr):.1f}°")

    gt, cs = None, [find_board(b) for b in bgrs]
    if cs[0] is not None and cs[1] is not None:
        if (cs[0][-1] - cs[0][0]) @ (cs[1][-1] - cs[1][0]) < 0:
            cs[1] = cs[1][::-1].copy()   # ponytail: 두 사진에서 코너 순서가 뒤집힌 경우. 카메라끼리 90° 넘게 돌아가 있으면 틀린 판단
        (Rb0, tb0, e0), (Rb1, tb1, e1) = solve_board(cs[0], Ks[0]), solve_board(cs[1], Ks[1])
        Rg, tg = relative(Rb0, tb0, Rb1, tb1)
        gt = Rg, center(Rg, tg), OBJP @ Rb0.T + tb0
        print(f"체커보드 정답: 두 카메라 사이 각도 {rot_deg(Rg):.1f}°, 거리 {1000 * np.linalg.norm(gt[1]):.0f}mm, "
              f"재투영 오차 {e0:.2f} / {e1:.2f}px")

    rows, shown = [], None
    for kname, Kp in (("calib", [scale_K(K, W, H) for K in Ks]), ("da3", list(p.intrinsics))):
        good = p.conf[0] >= np.percentile(p.conf[0], CONF_PCT)
        metric0 = raw * (Kp[0][0, 0] + Kp[0][1, 1]) / 2 / 300   # DA3METRIC 미터 변환 (focal은 처리 해상도 기준)
        s = float(np.median(metric0[good] / p.depth[0][good]))   # DA3-SMALL 단위 → 미터
        clouds = []
        for i in range(2):
            d = np.where(p.conf[i] >= np.percentile(p.conf[i], CONF_PCT), p.depth[i] * s, np.nan)
            P, _ = depth_to_points(d, Kp[i], p.processed_images[i], step=2, zmin=0, zmax=np.inf)
            clouds.append(to_world(P, Rs[i], ts[i] * s))
        Pc = clouds[1] @ Rs[0].T + ts[0] * s   # 이동 카메라 점 → 고정 카메라 좌표
        with np.errstate(divide="ignore", invalid="ignore"):
            u = Pc[:, 0] / Pc[:, 2] * Kp[0][0, 0] + Kp[0][0, 2]
            v = Pc[:, 1] / Pc[:, 2] * Kp[0][1, 1] + Kp[0][1, 2]
        inside = (Pc[:, 2] > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)   # 고정 카메라 시야 안 (가려짐은 무시)
        zmed = float(np.median(clouds[0][:, 2]))
        row = dict(tag=tag, K=kname, E0_rot=rot_deg(Rs[0]), cam_angle=rot_deg(Rr),
                   baseline_m=s * float(np.linalg.norm(C1)), s=s, overlap_frac=float(inside.mean()),
                   spacing_mm=1000 * zmed * 2 / Kp[0][0, 0])   # step=2라 점 간격은 2픽셀
        if inside.sum() >= 100:
            pc0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(clouds[0]))
            pc1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(clouds[1][inside]))
            dist = float(np.median(np.asarray(pc1.compute_point_cloud_distance(pc0))))
            row.update(overlap_med_mm=1000 * dist, overlap_pct=100 * dist / zmed)
        if gt is not None:
            Rg, Cg, corners_gt = gt
            cw = [to_world(unproject(cs[i] * [W / 640, H / 480], sample(p.depth[i], cs[i], W, H) * s, Kp[i]),
                           Rs[i], ts[i] * s) for i in range(2)]
            row.update(gt_angle=rot_deg(Rg), rot_err=rot_deg(Rr.T @ Rg), dir_err=angle_deg(C1, Cg),
                       baseline_ratio=row["baseline_m"] / float(np.linalg.norm(Cg)),
                       metric_ratio=float(np.median(sample(metric0, cs[0], W, H) / corners_gt[:, 2])),
                       corner3d_mm=1000 * float(np.linalg.norm(cw[0] - cw[1], axis=1).mean()),
                       corner_gt_mm=1000 * float(np.linalg.norm(cw[0] - corners_gt, axis=1).mean()))
        print("  " + "  ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items() if k != "tag"))
        rows.append(row)
        shown = shown or (clouds, s)   # 화면에는 K=calib 결과를 그린다
    save_rows(rows)

    if show:
        clouds, s = shown
        geoms = []
        for i, color in enumerate(((1, 0, 0), (0, 0, 1))):
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(clouds[i]))
            pc.paint_uniform_color(color)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Rs[i].T, center(Rs[i], ts[i] * s)
            geoms += [pc, o3d.geometry.TriangleMesh.create_coordinate_frame(0.05).transform(T)]
        for g in geoms:
            g.transform(FLIP)
        o3d.visualization.draw_geometries(geoms, window_name=f"{tag}: red=fixed blue=move")
    return rows


def selftest():
    rng = np.random.default_rng(0)
    R0, R1 = (cv2.Rodrigues(rng.normal(size=3))[0] for _ in range(2))
    t0, t1, X = rng.normal(size=3), rng.normal(size=3), rng.normal(size=(5, 3))
    Rr, tr = relative(R0, t0, R1, t1)
    assert np.allclose(X @ R1.T + t1, (X @ R0.T + t0) @ Rr.T + tr)   # 카메라0 좌표 → 카메라1 좌표
    assert np.allclose(to_world(X @ R0.T + t0, R0, t0), X)             # 카메라 → 월드 되돌리기
    assert np.allclose(center(R1, t1) @ R1.T + t1, 0)                  # 카메라 중심은 카메라 좌표의 원점
    assert abs(rot_deg(cv2.Rodrigues(np.array([0, 0, np.radians(37)]))[0]) - 37) < 1e-6
    assert abs(angle_deg(np.array([1.0, 0, 0]), np.array([1.0, 1, 0])) - 45) < 1e-6
    K = scale_K(np.array([[700, 0, 320], [0, 700, 240], [0, 0, 1.0]]), 504, 378)
    assert np.allclose(K, [[551.25, 0, 252], [0, 551.25, 189], [0, 0, 1]])
    assert np.allclose(unproject(np.array([[252 + 551.25, 189.0]]), np.array([2.0]), K), [[2, 0, 2]])
    print("two_view_check selftest OK")


def capture_loop(Ks, da3, metric, show):
    caps = [open_camera(FIX_CAM), open_camera(MOVE_CAM)]
    PAIRS.mkdir(exist_ok=True)
    while True:
        for c in caps:
            c.grab()   # 두 카메라를 최대한 같은 순간에
        frames = [c.retrieve()[1] for c in caps]
        cv2.imshow("fixed | move   (SPACE=capture, ESC=quit)", np.hstack(frames))
        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == 32:
            tag = time.strftime("%Y%m%d_%H%M%S")
            for name, f in zip(("fixed", "move"), frames):
                cv2.imwrite(str(PAIRS / f"{tag}_{name}.png"), f)
            evaluate(frames, Ks, da3, metric, tag, show)
    for c in caps:
        c.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cams", action="store_true", help="두 카메라로 찍으면서 평가")
    mode.add_argument("--pair", nargs=2, metavar=("FIXED_PNG", "MOVE_PNG"))
    mode.add_argument("--selftest", action="store_true")
    ap.add_argument("--calib-fixed", default=FIX_CALIB)
    ap.add_argument("--calib-move", default=MOVE_CALIB)
    ap.add_argument("--no-show", action="store_true", help="Open3D 창 안 띄움")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    from bench_da3 import set_precision   # DA3보다 먼저: DA3 로그 수준 환경변수를 여기서 정함
    from depth_anything_3.api import DepthAnything3

    set_precision("fp32")   # 이 그래픽카드에선 fp32가 가장 빠름 (bench_da3.txt)
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to("cuda").eval()
    metric = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda").eval()
    Ks = [np.load(args.calib_fixed)["K"], np.load(args.calib_move)["K"]]
    if args.pair:
        bgrs = [cv2.imread(p) for p in args.pair]
        assert all(b is not None and b.shape[:2] == (480, 640) for b in bgrs), "640x480 사진 두 장이 필요"
        evaluate(bgrs, Ks, da3, metric, Path(args.pair[0]).stem, not args.no_show)
    else:
        capture_loop(Ks, da3, metric, not args.no_show)


if __name__ == "__main__":
    main()
