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
small = cloud.voxel_down_sample(voxel_size=0.0001)
clean, kept = small.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
assert 0 < len(clean.points) <= len(small.points) <= len(cloud.points)
print("원본·축소·정리:", len(cloud.points), len(small.points), len(clean.points))

kdtree = o3d.geometry.KDTreeFlann(cloud)
query_point = cloud.points[0]
k, idx, dist = kdtree.search_knn_vector_3d(query_point, 10)

# neighbor_point = p[idx]
# neighbor_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(neighbor_point))

points = np.asarray(cloud.points)

neighbor_points = points[idx]

neighbor_cloud = o3d.geometry.PointCloud()
neighbor_cloud.points = o3d.utility.Vector3dVector(neighbor_points)

o3d.visualization.draw_geometries([neighbor_cloud])


# o3d.visualization.draw_geometries([neighbor_cloud])
