set -e
for i in {10..200..10}
do
	echo begin $i
	python track.py --load_iteration -1 -s temp_datasets/loquat-2 -m temp_datasets/loquat-1/output_debug --use_bounding_box --crop_by_mask --volume_init  --track_render_iterations $i
done

