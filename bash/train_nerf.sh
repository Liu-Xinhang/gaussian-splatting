set -e
for str in "chair" "drums" "ficus" "hotdog" "lego" "materials" "mic" "ship"; do
  	echo $str
	python train.py -s temp_datasets/nerf/nerf_synthetic/$str -m temp_datasets/nerf/nerf_synthetic/$str/output_debug --mytype nerf 
done
