set -e
for str in "fern" "fortress" "horns" "room"; do
  	echo $str
	python train.py -s temp_datasets/nerf/nerf_llff_data/$str -m temp_datasets/nerf/nerf_llff_data/$str/output_debug --mytype llff --eval
done
