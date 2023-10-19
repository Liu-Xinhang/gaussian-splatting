from plyfile import PlyData
import numpy as np
import math
from tqdm import tqdm
from scipy.spatial import ConvexHull
import random
import sys

def load_ply(ply_file):
	ply_data = PlyData.read(ply_file)

	# 访问点数据 (如果它是一个顶点数据)
	vertex_data = ply_data['vertex']
	x = vertex_data['x']
	y = vertex_data['y']
	z = vertex_data['z']

	# 或者将它转换为NumPy数组
	points = np.column_stack((x, y, z))
	return points


def calculate_models_diameter(vertices):
	"""
	This function will return the farthest distance between two points in the models, 
	which is mostly called the diameter of one model in the pose estimation task.

	Parameters
	----------
	vertices : ndarray
		The points of that model.

	Returns
	----------
	diameter : float
		The distance of the farthest two points.
	"""

	def calc_pts_diameter(vertices):
		"""计算模型的最大直径"""

		diameter = -1
		for pt_id in tqdm(range(vertices.shape[0])):
			# 将p.array([pts[pt_id, :]])在行与列分别重复pts.shape[0]和1次
			pt_dup = np.tile(np.array([vertices[pt_id, :]]), [vertices.shape[0] - pt_id, 1])  #
			pts_diff = pt_dup - vertices[pt_id:, :]
			max_dist = math.sqrt((pts_diff * pts_diff).sum(axis=1).max())
			if max_dist > diameter:
				diameter = max_dist
		return diameter

	hull = ConvexHull(vertices)
	valid_points = hull.points[np.unique(hull.simplices)]
	diameter = calc_pts_diameter(valid_points)
	
	return diameter


def distance(pose1, pose2):
	rot1 = pose1["pose"][:3, :3]
	rot2 = pose2["pose"][:3, :3]
	return np.arccos(np.clip((np.trace(rot1.T @ rot2) - 1)/2, -1, 1)) * 180 / np.pi

def select_poses(results, N, threshold):
	res = []
	while len(res) < N and results:
		sys.stdout.write('\r')
		# the exact output you're looking for:
		sys.stdout.write(f"res: {len(res)}/ results: {len(results)}")
		sys.stdout.flush()
	
		selected = random.choice(results)
		results.remove(selected)
		res.append(selected)
		results = [pose for pose in results if distance(pose, selected) > threshold]
	return res


if __name__ == "__main__":
	diameter = calculate_models_diameter(load_ply("temp_datasets/nerf/nerf_synthetic/hotdog/points3d.ply"))
	print(diameter)