from .general_utils import get_expon_lr_func

def get_constant_then_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, const_step=500000, max_steps=1000000):
	"""
	这个函数在get_expon_lr_func的基础上增加了一个const_step参数，用于在最开始的时候保持学习率不变，随后将使用expond_lr_func的学习率

	Returns
	-------
	_type_
		_description_
	"""
	assert const_step < max_steps, "const_step must be smaller than max_steps"

	origin_helper = get_expon_lr_func(lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps-const_step)
	def helper(step):
		if step < const_step:
			return lr_init
		return origin_helper(step-const_step)
	
	return helper

def get_constant_lr_func(lr):
	"""
	这是一个常数的学习率
	"""
	def helper(step):
		return lr
	return helper


def get_multi_step_lr_func(lr_init, milestones=[30,80], gamma=0.1):
	def helper(step):
		for i in range(len(milestones)):
			if step < milestones[i]:
				return lr_init * gamma ** i
		return lr_init * gamma ** len(milestones)

	return helper
