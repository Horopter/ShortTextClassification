counter=1
array=()
until [ $counter -gt 32 ]
do
	eval "a= python3 -B TextClassifier.py | grep Final"
	echo $a
	array+=($a)
	((counter++))
done
echo $array
