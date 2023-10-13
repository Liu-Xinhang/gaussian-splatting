def get_multi_step_lr_func(lr_init, milestones=[30,80], gamma=0.1):
	def helper(step):
		for i in range(len(milestones)):
			if step < milestones[i]:
				return lr_init * gamma ** i
		return lr_init * gamma ** len(milestones)

	return helper

if __name__ == "__main__":
	lr_func = get_multi_step_lr_func(1, [30,80], 0.1)
	for i in range(100):
		print(i, lr_func(i))