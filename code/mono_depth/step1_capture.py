"""Step 1: 웹캠으로 사진 한 장 찍기. 스페이스 = 저장, ESC = 종료. (--auto: 30프레임 뒤 자동 저장)"""
import sys

import cv2

from common import HERE, open_source

cap = open_source(sys.argv)
n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imshow("SPACE: save test.jpg / ESC: quit", frame)
    k = cv2.waitKey(1) & 0xFF
    n += 1
    if k == ord(" ") or ("--auto" in sys.argv and n == 30):
        cv2.imwrite(str(HERE / "test.jpg"), frame)
        print("saved test.jpg", frame.shape)
        if "--auto" in sys.argv:
            break
    elif k == 27:
        break
cap.release()
cv2.destroyAllWindows()
