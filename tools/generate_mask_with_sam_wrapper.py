import subprocess
from pathlib import Path

if __name__ == "__main__":
	## grab all data
	skip_times =  1
	current_iter = 0
	root_path = Path("temp_datasets/onepose_datasets")
	total_objects = list(sorted(root_path.glob("*")))
	for obj in total_objects:
		total_sequence = list(sorted(obj.glob("*")))
		for subdir in total_sequence:
			if not subdir.is_dir():
				continue
			if current_iter < skip_times:
				current_iter += 1
				continue
			if not (subdir / "input").exists():
				print(f"begin export png from video {subdir.stem}")
				subprocess.run(f"python /home/liuxinhang/Projects/gaussian-splatting/tools/convert_video_to_images.py --root_dir {str(subdir)}", shell=True)
			else:
				print(f"skip export png from video {subdir.stem}")
			print(f"Processing {str(subdir)}")
			subprocess.run(f"python /home/liuxinhang/Projects/gaussian-splatting/tools/generate_mask_with_sam.py --root_dir {str(subdir)}", shell=True)
