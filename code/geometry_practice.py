import numpy as np

R = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
t = np.array([0.4, 0.1, 0.3])
T = np.eye(4)
T[:3, :3], T[:3, 3] = R, t
p_camera = np.array([0.1, 0.0, 0.5, 1.0])

p_base = T @ p_camera
assert np.allclose(R.T @ R, np.eye(3))
assert np.isclose(np.linalg.det(R), 1.0)
assert np.allclose(p_base[:3], [0.4, 0.2, 0.8])
assert np.allclose(np.linalg.inv(T) @ p_base, p_camera)
print("바닥 기준 좌표:", p_base[:3])
