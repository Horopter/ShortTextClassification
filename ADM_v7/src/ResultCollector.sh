counter=1
array=()
until [ $counter -gt 201 ]
do
	eval "a= python3 -B TextClassifier.py $((counter)) | grep Final"
	echo $a
	array+=($a)
	((counter++))
done
echo $array
