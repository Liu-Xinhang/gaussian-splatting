set -e
str_array=("/home/liuxinhang/Projects/gaussian-splatting/temp_datasets/onepose_datasets/0601-loquat-box/loquat-1"
"/home/liuxinhang/Projects/gaussian-splatting/temp_datasets/onepose_datasets/0602-aficion-box/aficion-1"
"/home/liuxinhang/Projects/gaussian-splatting/temp_datasets/onepose_datasets/0603-redbook-box/redbook-1"
"/home/liuxinhang/Projects/gaussian-splatting/temp_datasets/onepose_datasets/0604-pillbox1-box/pillbox1-1"
"/home/liuxinhang/Projects/gaussian-splatting/temp_datasets/onepose_datasets/0606-tiger-others/tiger-1")

for path in "${str_array[@]}"; do
    echo "$path"
	python train.py -s $path -m $path/output_debug --mytype Onepose --crop_by_mask --volume_init 
done