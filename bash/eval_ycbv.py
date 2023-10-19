import subprocess
import json
if __name__ == "__main__":
	with open("val.json") as file:
		data = json.load(file)

	for k, vs in data.items():
		for v in vs:
			print(k, v)
			subprocess.run(f"python /home/liuxinhang/Projects/gaussian-splatting/eval_ycb_bak.py --load_iteration -1 -s temp_datasets/ycbv/test/{int(v):06d} -m reference_images/{int(k):06d}/output_debug --mytype YCBV", shell=True)