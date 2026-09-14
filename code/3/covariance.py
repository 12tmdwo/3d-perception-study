import numpy as np


import numpy as np

rng = np.random.default_rng(7)
print(rng)
p = rng.normal(size=(3000, 3)) * [0.10, 0.02, 0.005]
print(p.shape)
angle = np.deg2rad(30)
R = np.array([[np.cos(angle), -np.sin(angle), 0],
              [np.sin(angle),  np.cos(angle), 0], 
              [0, 0, 1]])
p = p @ R.T
centered = p - p.mean(axis=0)
C = centered.T @ centered / len(p)
values, vectors = np.linalg.eigh(C)
main_axis = vectors[:, -1]
expected = np.array([np.cos(angle), np.sin(angle), 0])
assert abs(main_axis @ expected) > 0.99
print("주축:", main_axis)
print("작은 순서의 고유값:", values)


import open3d as o3d

cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
small = cloud.voxel_down_sample(voxel_size=0.005
)
clean, kept = small.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
assert 0 < len(clean.points) <= len(small.points) <= len(cloud.points)
print("원본·축소·정리:", len(cloud.points), len(small.points), len(clean.points))
# o3d.io.write_point_cloud("practice_cloud(clean_0.005).ply", clean)
# o3d.visualization.draw_geometries([clean])



# 각 행이 한 점입니다. 단위는 cm입니다.
points = np.array(clean.points)
center = points.mean(axis=0)
difference = points - center
covariance = difference.T @ difference / len(points)

# 대칭행렬의 고유값은 작은 순서로 반환됩니다.
values, axes = np.linalg.eigh(covariance)
long_axis = axes[:, -1]
expected_axis = np.array([1., 1.]) / np.sqrt(2)

# assert np.allclose(center, [5, 5])
# assert np.allclose(covariance, [[5, 3], [3, 5]])
# assert np.allclose(values, [2, 8])
# assert np.isclose(abs(long_axis @ expected_axis), 1.0)
print("평균점:", center)
print("공분산:\n", covariance)
print("작은 순서의 고유값:", values)
print("긴 축:", long_axis)


# 주축 3개를 선으로 그리기. 길이는 퍼진 정도(표준편차)에 비례시킵니다.
lengths = 3 * np.sqrt(values)
line_points = [center]
lines = []
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 작은 축→빨강, 중간→초록, 긴 축→파랑
for i in range(3):
    line_points.append(center + axes[:, i] * lengths[i])
    lines.append([0, i + 1])  # 중심점(0)에서 각 축 끝점으로

axis_lines = o3d.geometry.LineSet(
    o3d.utility.Vector3dVector(line_points),
    o3d.utility.Vector2iVector(lines),
)
axis_lines.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([clean, axis_lines])

