set -e
for i in {1..21..1};do
	formatted_number=$(printf "%06d" $i)
	echo "handle $i"
	python train.py -s reference_images/$formatted_number -m reference_images/$formatted_number/output_debug --mytype YCBVRender
done