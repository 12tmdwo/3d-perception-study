"""1·2단계: 이동 카메라 사진마다 위치를 구해 물체 점을 고정 카메라 좌표에 쌓는다.

위치 구하기(키프레임 이어붙이기): 첫 프레임만 [고정 카메라 기준 사진, 지금 사진]으로 좌표계를 잡고,
그 뒤로는 [최근 키프레임, 지금 사진]으로 이어 붙인다. 15° 넘게 돌거나 15cm 넘게 가면 지금 사진이 새 키프레임.
(고정 카메라 사진만 기준으로 쓰면 방향 차이 ~80°에서 회전이 4~5° 흔들리고 가끔 뒤집혔다 — 2026-09-14 녹화 분석)
쌓기: 새 물체 점을 이미 쌓인 점에 ICP(이동·회전·크기)로 맞춘 뒤 쌓는다. 겹침이 적거나 보정이 크면 그 프레임은 버린다.
(DA3는 짝지어 넣는 사진에 따라 같은 사진의 깊이 모양이 달라져 겹이 어긋났다 — 2026-09-14 측정. 첫 프레임 점은 그래서 안 쌓음)

실행: cd code/scan && ~/venvs/da3/bin/python scan_live.py [--cls cup] [--erode 10] [--da3-k]
      처음 뜨는 고정 카메라 창에서 SPACE = 기준 사진(사람 없이, 스캔할 물체는 보이게). 그다음 ESC = 종료.
녹화로:   --ref 세션/ref.png --image 세션/move_%04d.jpg --frames 400
사진으로: --ref pairs/..._fixed.png --image pairs/..._move.png --frames 10        자체 검사: python3 scan_live.py --selftest
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mono_depth"))
from common import FLIP, ImageCap, depth_to_points  # noqa: E402

# 두 C270은 일련번호가 같아 by-id로 구별이 안 됨 → 꽂힌 USB 자리(by-path)로 연다. 자리를 바꾸면 역할이 뒤바뀐다.
MOVE_CAM = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3:1.0-video-index0"
MOVE_CALIB = HERE.parent / "calibration" / "camera_params.npz"
FIX_CAM = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:1.1:1.0-video-index0"   # 고정 카메라 (USB 1.1 자리, 모니터 위)
FIX_CALIB = HERE.parent / "calibration" / "camera_params1.npz"
YOLO_PATH = HERE.parents[1] / "yolo26n-seg.engine"
PERSON = "person"
CONF_PCT = 50             # 임시 기준: 기준 사진에서 DA3 신뢰도 상위 절반 픽셀로 눈금 계산
S_TOL = 0.10              # 임시 기준: 눈금이 최근 중앙값에서 10% 넘게 벗어나면 그 회는 버림 (키프레임 바뀌면 기록 비움)
VOXEL = 0.003             # 임시 기준: 쌓은 점을 3mm 칸마다 하나로 솎음
KF_DEG, KF_M = 15, 0.15   # 임시 기준: 키프레임에서 이만큼 돌거나 움직이면 새 키프레임 (녹화 35초에 13~20장)
ICP_CORR = 0.03                                      # 임시 기준: ICP가 3cm 안에서 짝 점을 찾음
ICP_MIN_FIT, ICP_MAX_MM, ICP_MAX_DEG = 0.3, 30, 10   # 임시 기준: 겹침이 이보다 적거나 보정이 이보다 크면 그 프레임 점은 버림
F4 = np.array(FLIP, float)                           # 화면용 뒤집기 (자기 자신이 역행렬)


def pick_mask(masks, classes, names, cls_name=None, erode_px=4):
    """(N,H,W) 마스크 중 대상 하나 고르기 → 사람(손) 영역 빼기 → 가장자리 깎기. (번호, bool 마스크) 또는 None.

    cls_name을 주면 그 클래스 중 가장 큰 것, 없으면 사람을 뺀 물건 중 가장 큰 것.
    깎는 이유: 경계 픽셀의 깊이는 뒤쪽 배경 값이 섞여 있어 점이 뒤로 늘어진다.
    """
    if len(masks) == 0:
        return None
    masks = np.asarray(masks) > 0.5
    labels = np.array([names[int(c)] for c in classes])
    cand = labels == cls_name if cls_name else labels != PERSON
    if not cand.any():
        return None
    i = int((masks.sum(axis=(1, 2)) * cand).argmax())
    m = masks[i].copy()
    if cls_name != PERSON:
        m &= ~masks[labels == PERSON].any(axis=0)
    if erode_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2)
        m = cv2.erode(m.astype(np.uint8), k).astype(bool)
    return (i, m) if m.any() else None


def rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def center(R, t):
    """'월드→카메라' 변환(X_cam = R X_w + t)의 카메라 중심 = 월드 좌표."""
    return -R.T @ t


def to_world(P, R, t):
    """카메라 좌표 점 (N,3) → 월드 좌표."""
    return (P - t) @ R


def compose(Rr, tr, R_kf, t_kf):
    """'월드→키프레임'과 '키프레임→지금 카메라'를 이어 '월드→지금 카메라'."""
    return Rr @ R_kf, Rr @ t_kf + tr


def scale_K(K, W, H):
    """640x480 기준 K → DA3 처리 해상도(W,H) 기준 K."""
    return K * np.array([[W / 640], [H / 480], [1]])


def scale_ok(s, recent):
    """눈금 s가 최근 값들의 중앙값에서 S_TOL 이내인가. 기록이 없으면 통과."""
    return len(recent) == 0 or abs(s / np.median(recent) - 1) <= S_TOL


def align_to_acc(src, dst):
    """Open3D 점구름 src를 이미 쌓인 점구름 dst에 ICP(이동·회전·크기)로 맞춘다.
    반환: 4x4 T (x' = T x, 크기 포함), 겹침 비율, 평균 점 이동(mm), 회전(°), 크기 차(%)"""
    import open3d as o3d
    reg = o3d.pipelines.registration.registration_icp(
        src, dst, ICP_CORR, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=True),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    T = reg.transformation
    sc = float(np.cbrt(np.linalg.det(T[:3, :3])))
    P = np.asarray(src.points)
    move_mm = 1000 * float(np.linalg.norm(P @ T[:3, :3].T + T[:3, 3] - P, axis=1).mean())
    return T, reg.fitness, move_mm, rot_deg(T[:3, :3] / sc), 100 * (sc - 1)


def correct_pose(R, t, T):
    """월드 점을 x' = T x 로 보정했을 때 같은 카메라의 '월드→카메라' 자세. 크기는 빼고 회전·이동만 반영한다."""
    A = T[:3, :3] / np.cbrt(np.linalg.det(T[:3, :3]))
    R2 = R @ A.T
    return R2, t - R2 @ T[:3, 3]


def selftest():
    import open3d as o3d

    names = {0: PERSON, 39: "bottle", 41: "cup"}
    masks = np.zeros((3, 40, 60), np.float32)
    masks[0, 5:15, 5:15] = 1      # 작은 컵
    masks[1, 10:35, 20:50] = 1    # 큰 컵
    masks[2, 10:35, 40:60] = 1    # 손이 큰 컵 오른쪽을 가림
    cls = [41, 41, 0]
    i, m = pick_mask(masks, cls, names, "cup", erode_px=0)
    assert i == 1 and m.sum() == 25 * 20 and not m[:, 40:].any()   # 큰 컵, 손 부분은 빠짐
    i, m2 = pick_mask(masks, cls, names, None, erode_px=2)
    assert i == 1 and 0 < m2.sum() < m.sum()                        # 이름 없으면 사람 빼고 가장 큰 것, 깎여서 줄어듦
    assert pick_mask(masks, cls, names, "bottle") is None
    assert pick_mask(masks[:0], [], names) is None

    assert scale_ok(1.3, []) and scale_ok(1.05, [1.0, 1.0, 1.1]) and not scale_ok(1.2, [1.0, 1.0, 1.1])
    R = cv2.Rodrigues(np.array([0.1, -0.4, 0.2]))[0]
    t, X = np.array([0.3, -0.1, 0.5]), np.random.default_rng(0).normal(size=(5, 3))
    assert np.allclose(to_world(X @ R.T + t, R, t), X) and np.allclose(center(R, t) @ R.T + t, 0)
    assert abs(rot_deg(cv2.Rodrigues(np.array([0, 0, np.radians(37)]))[0]) - 37) < 1e-6
    R2, t2 = cv2.Rodrigues(np.array([-0.3, 0.2, 0.05]))[0], np.array([0.05, 0.1, -0.2])
    Rc, tc = compose(R2, t2, R, t)
    assert np.allclose((X @ R.T + t) @ R2.T + t2, X @ Rc.T + tc)   # 월드→키프레임→지금 = 월드→지금

    A, b = cv2.Rodrigues(np.array([0.05, 0.02, -0.03]))[0], np.array([0.01, -0.02, 0.005])
    Tb = np.eye(4)
    Tb[:3, :3], Tb[:3, 3] = A, b
    R3, t3 = correct_pose(R, t, Tb)
    assert np.allclose((X @ A.T + b) @ R3.T + t3, X @ R.T + t)   # 보정한 월드 점을 새 자세로 보면 원래 카메라 좌표

    g = np.arange(-0.1, 0.1, 0.004)
    xx, yy = np.meshgrid(g, g)
    surf = np.column_stack((xx.ravel(), yy.ravel(), (0.4 + 0.02 * np.sin(30 * xx) * np.cos(30 * yy)).ravel()))   # 울퉁불퉁한 면
    Tt = np.eye(4)
    Tt[:3, :3], Tt[:3, 3] = 1.03 * cv2.Rodrigues(np.array([0, 0, np.radians(5)]))[0], [0.01, 0, 0]
    moved = (surf - Tt[:3, 3]) @ np.linalg.inv(Tt[:3, :3]).T   # Tt를 거꾸로 적용 → ICP로 맞추면 Tt가 나와야 함
    pc = lambda P: o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    T, fit, mm, deg, pct = align_to_acc(pc(moved), pc(surf))
    assert fit > 0.9 and abs(deg - 5) < 1 and abs(pct - 3) < 1, (fit, mm, deg, pct)   # 5° 돌리고 3% 키운 만큼 되돌림
    print("scan_live selftest OK")


def open_camera(src):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ok, frame = cap.read()
    assert ok and frame.shape[:2] == (480, 640), f"카메라를 못 열었거나 640x480이 아님: {src} (USB 자리 확인)"
    return cap


def capture_ref(src):
    """고정 카메라 미리보기 → SPACE로 기준 사진 한 장. ESC면 None."""
    cap = open_camera(src)
    while True:
        frame = cap.read()[1]
        cv2.imshow("fixed camera: SPACE = reference (no people), ESC = quit", frame)
        key = cv2.waitKey(1)
        if key in (27, 32):
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame if key == 32 else None


def main():
    dev = lambda s: int(s) if str(s).isdigit() else s
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", default=MOVE_CAM, help="이동 카메라 번호")
    ap.add_argument("--calib", default=MOVE_CALIB, help="이동 카메라 캘리브레이션 파일 설정")
    ap.add_argument("--fix-cam", default=FIX_CAM, help="고정 카메라 번호")
    ap.add_argument("--fix-calib", default=FIX_CALIB, help="고정 카메라 캘리브레이션")
    ap.add_argument("--ref", help="고정 카메라 대신 기준 사진 파일")
    ap.add_argument("--image", help="이동 카메라 대신: 사진 한 장(계속 반복) 또는 녹화 파일 형식(예: 세션/move_%%04d.jpg)")
    ap.add_argument("--cls", help="대상 클래스 이름 (예: cup). 없으면 사람 빼고 가장 큰 물건")
    ap.add_argument("--erode", type=int, default=10, help="마스크 가장자리 깎는 픽셀 수")
    ap.add_argument("--yolo", default=YOLO_PATH)
    ap.add_argument("--da3-k", action="store_true", help="캘리브레이션 대신 DA3 추정 초점거리로 계산 (비교용)")
    ap.add_argument("--frames", type=int, default=0, help="이 프레임 수만큼 돌고 종료 (0 = ESC까지)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    from bench_da3 import set_precision   # DA3보다 먼저: DA3 로그 수준 환경변수를 여기서 정함
    import open3d as o3d
    from depth_anything_3.api import DepthAnything3
    from ultralytics import YOLO

    K_fix, K_move = np.load(args.fix_calib)["K"], np.load(args.calib)["K"]
    yolo = YOLO(str(args.yolo), task="segment")
    # 빈 사진으로 한 번 돌려 이름표를 얻는다 (yolo.names를 먼저 읽으면 엔진을 한 번 더 불러와 메모리를 이중으로 씀)
    names = yolo.predict(np.zeros((480, 640, 3), np.uint8), imgsz=640, retina_masks=True, verbose=False)[0].names
    if args.cls and args.cls not in names.values():
        sys.exit(f"--cls {args.cls}: YOLO 클래스에 없음. 가능한 이름: {sorted(names.values())}")

    # 기준 사진: 좌표계(월드 = 고정 카메라)를 잡는 데 첫 프레임에서만 쓴다
    ref = cv2.imread(args.ref) if args.ref else capture_ref(dev(args.fix_cam))
    if ref is None:
        sys.exit("기준 사진 없이 종료")
    assert ref.shape[:2] == (480, 640), "캘리브레이션 K는 640x480 기준"
    if not args.ref:
        (HERE / "pairs").mkdir(exist_ok=True)
        cv2.imwrite(str(HERE / "pairs" / f"ref_{time.strftime('%Y%m%d_%H%M%S')}.png"), ref)
    ref_rgb = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
    set_precision("fp32")   # 이 그래픽카드에선 fp32가 가장 빠름 (bench_da3.txt)
    metric = DepthAnything3.from_pretrained("depth-anything/DA3METRIC-LARGE").to("cuda").eval()   # 기준 사진·키프레임 미터 깊이
    raw_ref = metric.inference([ref_rgb], process_res=504).depth[0]   # 미터 = raw × focal / 300
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL").to("cuda").eval()
    if args.image:
        cap = cv2.VideoCapture(args.image) if "%" in args.image else ImageCap(args.image)
    else:
        cap = open_camera(dev(args.cam))

    vis = o3d.visualization.Visualizer()
    assert vis.create_window("Object Points", 640, 480), "3D 창을 못 열었다 → 터미널에서 glxinfo -B 로 드라이버 확인"
    acc, path = o3d.geometry.PointCloud(), o3d.geometry.LineSet()
    move_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(0.05).transform(FLIP))   # 고정 카메라 = 원점
    added, n, used, t0 = False, 0, 0, time.time()
    scales, centers, rots, track = deque(maxlen=10), deque(maxlen=10), deque(maxlen=10), []
    kf, nkf = None, 0   # 키프레임 = (RGB, DA3METRIC 원시 깊이, 월드→키프레임 R, t)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        assert frame.shape[:2] == (480, 640), "캘리브레이션 K는 640x480 기준"
        # frame = cv2.undistort(frame, K, dist)   # 사용자가 끔: 기존 dist가 이 C270 가장자리를 과보정
        r = yolo.predict(frame, imgsz=640, retina_masks=True, verbose=False)[0]
        picked = None
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy()
            assert masks.shape[1:] == frame.shape[:2], f"마스크 크기 {masks.shape} ≠ 사진 크기"
            picked = pick_mask(masks, r.boxes.cls.cpu().numpy(), r.names, args.cls, args.erode)

        view, info, info2 = frame.copy(), "no target", ""
        if picked is not None:
            i, mask = picked
            cur = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # ponytail: 첫 위치 한 번이 좌표계 기준이라 그 오차(±2~3cm, 4~5°)는 모든 위치에 같은 만큼 남는다. 모양 복원엔 영향 없음
            base_rgb, base_raw, base_K = (ref_rgb, raw_ref, K_fix) if kf is None else (kf[0], kf[1], K_move)
            p = da3.inference([base_rgb, cur], process_res=504, ref_view_strategy="first")
            H, W = p.depth.shape[1:]
            assert base_raw.shape == (H, W), (base_raw.shape, H, W)
            K0, K1 = (p.intrinsics[0], p.intrinsics[1]) if args.da3_k else (scale_K(base_K, W, H), scale_K(K_move, W, H))
            good = p.conf[0] >= np.percentile(p.conf[0], CONF_PCT)
            s = float(np.median(base_raw[good] * (K0[0, 0] + K0[1, 1]) / 600 / p.depth[0][good]))   # DA3 단위 → 미터
            view[mask] = (view[mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
            if not scale_ok(s, scales):
                info = f"skip: s {s:.2f} vs recent {np.median(scales):.2f}"
            else:
                scales.append(s)
                Rr, tr = p.extrinsics[1, :, :3], p.extrinsics[1, :, 3] * s   # 기준(고정 카메라 또는 키프레임) → 지금 카메라
                R, t = (Rr, tr) if kf is None else compose(Rr, tr, kf[2], kf[3])   # 월드(고정 카메라) → 지금 카메라
                small = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                pts, cols = depth_to_points(np.where(small, p.depth[1] * s, np.nan), K1, p.processed_images[1], zmin=0, zmax=np.inf)
                new = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(to_world(pts, R, t)))
                new.colors = o3d.utility.Vector3dVector(cols)
                new = new.transform(F4).voxel_down_sample(VOXEL)   # acc와 같은 화면 좌표로
                if kf is None:
                    # 짝이 고정 카메라(방향 차 ~80°)면 깊이 모양이 ~20° 뒤틀린다 → 좌표계만 잡고 점은 안 쌓음
                    keep, info2 = False, "first frame: sets world, points not added"
                elif len(acc.points) == 0:
                    keep, info2 = True, "icp - (first layer)"
                else:
                    T, fit, mm, deg, pct = align_to_acc(new, acc)
                    keep = fit >= ICP_MIN_FIT and mm <= ICP_MAX_MM and deg <= ICP_MAX_DEG
                    info2 = f"icp {mm:.0f}mm {deg:.1f}deg scale {pct:+.1f}% fit {fit:.2f}" + ("" if keep else "  REJECT")
                    if keep:
                        new.transform(T)
                        R, t = correct_pose(R, t, F4 @ T @ F4)   # 화면 좌표 보정 → 월드 좌표 보정으로 바꿔 위치에 반영
                if keep:
                    acc += new
                    used += 1
                    if used % 5 == 0:   # 쌓인 점 솎기 (vis가 acc 객체를 들고 있어서 내용만 바꿔 넣는다)
                        down = acc.voxel_down_sample(VOXEL)
                        acc.points, acc.colors = down.points, down.colors
                if kf is None or (keep and (rot_deg(Rr) > KF_DEG or np.linalg.norm(tr) > KF_M)):   # ICP에서 버린 프레임은 키프레임 안 됨
                    kf = (cur, metric.inference([cur], process_res=504).depth[0], R, t)
                    scales.clear()   # 키프레임이 바뀌면 DA3 단위도 바뀐다
                    nkf += 1

                C = center(R, t)
                centers.append(C)
                rots.append(R)
                track.append(C * [1, -1, -1])   # FLIP 적용
                Tc = np.eye(4)
                Tc[:3, :3], Tc[:3, 3] = R.T, C
                axes = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05).transform(Tc).transform(FLIP)
                move_axes.vertices, move_axes.vertex_normals = axes.vertices, axes.vertex_normals
                path.points = o3d.utility.Vector3dVector(track)
                path.lines = o3d.utility.Vector2iVector([[k, k + 1] for k in range(len(track) - 1)])
                if len(acc.points):
                    if not added:   # 물체 점이 처음 쌓인 프레임에 창 시점을 맞춘다 (빈 점구름을 올리면 시점이 엉뚱해짐)
                        vis.add_geometry(acc)
                        vis.add_geometry(move_axes)
                        added = True
                    else:
                        vis.update_geometry(acc)
                        vis.update_geometry(move_axes)
                if len(track) == 2:   # 경로 선은 점이 두 개 생긴 뒤에 올린다 (빈 선을 올리면 Open3D 경고)
                    vis.add_geometry(path, reset_bounding_box=False)
                elif len(track) > 2:
                    vis.update_geometry(path)
                jit_mm = 1000 * float(np.linalg.norm(np.std(centers, axis=0)))
                jit_deg = max(rot_deg(Rk @ R.T) for Rk in rots)
                info = (f"{r.names[int(r.boxes.cls[i])]} s={s:.2f} cam=({C[0]*100:.0f},{C[1]*100:.0f},{C[2]*100:.0f})cm "
                        f"jitter {jit_mm:.1f}mm {jit_deg:.2f}deg kf {nkf} pts {len(acc.points)}")
        vis.poll_events()
        vis.update_renderer()
        cv2.putText(view, info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(view, info2, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imshow("Mask", view)
        n += 1
        if n % 10 == 0:
            print(f"FPS {n / (time.time() - t0):.1f} | {info} | {info2}", flush=True)
        if cv2.waitKey(1) == 27 or n == args.frames:
            break
    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()


if __name__ == "__main__":
    main()
