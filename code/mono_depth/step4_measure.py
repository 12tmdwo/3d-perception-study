"""Step 4: 화면을 클릭하면 그 픽셀까지의 거리(m)를 보여준다. ESC = 종료.
카메라 없으면: --image code/mono_depth/test.jpg   자동 종료: --frames 30"""
import sys

import cv2

from common import load_camera, open_source, load_depth_model, predict_depth

K, dist = load_camera()
pipe = load_depth_model(metric=True)
cap = open_source(sys.argv)
frames = int(sys.argv[sys.argv.index("--frames") + 1]) if "--frames" in sys.argv else 0

click = [320, 240]   # 클릭한 픽셀. 처음엔 화면 중앙


def on_mouse(event, x, y, *_):
    if event == cv2.EVENT_LBUTTONDOWN:
        click[:] = [x, y]


cv2.namedWindow("click = distance")
cv2.setMouseCallback("click = distance", on_mouse)
n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.undistort(frame, K, dist)
    depth = predict_depth(pipe, frame)
    x, y = click
    z = depth[y, x]
    cv2.circle(frame, (x, y), 5, (0, 255, 0), 2)
    cv2.putText(frame, f"{z:.2f} m", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("click = distance", frame)
    n += 1
    if n % 10 == 0:
        print(f"pixel ({x},{y}) depth = {z:.2f} m")
    if cv2.waitKey(1) == 27 or n == frames:
        break
cap.release()
cv2.destroyAllWindows()
