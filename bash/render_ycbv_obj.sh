set -e
for i in {1..20..1};do
	echo "Rendering object $i"
	python tools/render_ycbv_images.py --id $i
done