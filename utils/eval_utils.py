import numpy as np
from scipy.spatial.transform import Rotation as R

class DegreeAndCM():
	"""
	The loss about the degree and the cm. like 5 degree and 5 cm.
	"""
	z_axis = np.array([0, 0, -1], dtype=np.float32).reshape(-1, 1)

	def __init__(self, translation_scale, seperate_axis=False) -> None:

		self.translation_scale = translation_scale ## onepose dataset is m
		self._seperate_axis = seperate_axis
		self.reset()

	def _compute_the_degree_and_centimeter(self, pose_gt, pose_pred, translation_scale):
		"""
		Compute the degree and the centimeter.
		"""
		
		trans_gt = pose_gt[..., :3, 3] # L * 3
		trans_pred = pose_pred[..., :3, 3] # L * 3
		rot_gt = pose_gt[..., :3, :3] # L * 3 * 3
		rot_pred = pose_pred[..., :3, :3] # L * 3 * 3
		trans_distance = np.linalg.norm(trans_gt - trans_pred, axis=-1) # L,
		value = 0.5 * (np.trace(rot_gt @ np.moveaxis(rot_pred, -1, -2), axis1=-1, axis2=-2) - 1)
		value = np.clip(value, -1, 1)
		rot_distance = np.arccos(value)
		rot_distance = rot_distance / np.pi * 180 # convert to degree

		if self._seperate_axis:
			trans_distance_l1 = np.abs(trans_gt-trans_pred) # L * 3

		if translation_scale == "m":
			trans_distance *= 100
			if self._seperate_axis:
				trans_distance_l1 *= 100
		elif translation_scale == "mm":
			trans_distance /= 10
			if self._seperate_axis:
				trans_distance_l1 /= 10
		elif translation_scale == "cm":
			pass
		else:
			raise NotImplementedError("The translation scale is not implemented.")

		if self._seperate_axis:
			self._total_x.extend([trans_distance_l1[0]]) #! now it only support the batch size is 1
			self._total_y.extend([trans_distance_l1[1]])
			self._total_z.extend([trans_distance_l1[2]])

		return rot_distance, trans_distance
	
	def reset(self):
		self._total_degree = []
		self._total_centimeter = []
		if self._seperate_axis:
			self._total_x = []
			self._total_y = []
			self._total_z = []
		self.count = 0
	
	@staticmethod
	def omit_z_direction(pose_gt, pose_pr):
		"""
		返回去除掉z方向的位姿
		"""
		pose_gt_euler = R.from_matrix(pose_gt[:3, :3]).as_euler("ZYX")
		pose_gt_euler[0] = 0 ## 将Z方向固定为0
		pose_pred_euler = R.from_matrix(pose_pr[:3, :3]).as_euler("ZYX")
		pose_pred_euler[0] = 0
		return R.from_euler("ZYX", pose_gt_euler).as_matrix(), R.from_euler("ZYX", pose_pred_euler).as_matrix()
	
	def update(self, pose_gt, pose_pred, isValid=True, omit_z_direction=False):
		if omit_z_direction:
			pose_gt[:3, :3], pose_pred[:3, :3] = self.omit_z_direction(pose_gt, pose_pred)
			# print(pose_gt, pose_pred)
		if not isValid:
			rot_distance = np.nan
			trans_distance = np.nan
		else:
			rot_distance, trans_distance = self._compute_the_degree_and_centimeter(pose_gt, pose_pred, self.translation_scale)
		if np.isscalar(rot_distance):
			rot_distance = [rot_distance]
			trans_distance = [trans_distance]
		self._total_degree.extend(rot_distance)
		self._total_centimeter.extend(trans_distance)
		self.count += 1
	
	def get_current_degree_cm(self):
		return self._total_degree[-1], self._total_centimeter[-1]
	
	def get_the_ratio_of_degree_and_cm(self, degree, cm, combined_two=True):
		"""
		Get the ratio of the degree and the cm.
		Parameters
		----------
		degree : float
			the threshold of degree
		cm : float
			the threshold of cm
		combined_two : bool, optional
			whether bind the two together, if True, the return is one number. And if False, the return has two number, by default True
		"""

		degree_satisfy = np.array(self._total_degree) < degree 
		cm_satisfy = np.array(self._total_centimeter) < cm
		if combined_two:
			score = np.bitwise_and(degree_satisfy, cm_satisfy)
			return np.mean(score)
		return np.mean(degree_satisfy), np.mean(cm_satisfy)

	def get_the_mean_of_degree_and_cm(self):
		return np.mean(self._total_degree), np.mean(self._total_centimeter)
	
	def get_total_degree_and_cm(self):
		return np.array(self._total_degree), np.array(self._total_centimeter)
	
	def get_total_xyz(self):
		if not self._seperate_axis:
			raise ValueError("The seperate axis is not set.")
		return np.array(self._total_x), np.array(self._total_y), np.array(self._total_z)
