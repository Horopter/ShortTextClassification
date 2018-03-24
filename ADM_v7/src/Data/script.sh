for file in *.txt
do
	echo $file
	sep='dataChunk_'
	array="${file//$sep/$'_'}"
	imm="Rep_test_${array[0]}_${array2[0]}"
	mv "$file" "$imm"
done
for f in *.txt_; do 
mv -- "$f" "${f%.txt_}.chunk"
done