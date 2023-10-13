set -e
str_array=("/home/liuxinhang/Projects/gaussian-splatting/temp_datasets/onepose_datasets/0601-loquat-box/loquat-"
"/home/liuxinhang/Projects/gaussian-splatting/temp_datasets/onepose_datasets/0602-aficion-box/aficion-"
"/home/liuxinhang/Projects/gaussian-splatting/temp_datasets/onepose_datasets/0603-redbook-box/redbook-"
"/home/liuxinhang/Projects/gaussian-splatting/temp_datasets/onepose_datasets/0604-pillbox1-box/pillbox1-"
"/home/liuxinhang/Projects/gaussian-splatting/temp_datasets/onepose_datasets/0606-tiger-others/tiger-")

for path in "${str_array[@]}"; do
    echo "$path"
	python track.py --load_iteration -1 -s ${path}2 -m ${path}1/output_debug --crop_by_mask --volume_init
done