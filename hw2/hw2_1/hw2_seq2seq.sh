#!/bin/bash
if ! [  -f ./modelBest-Current -a  -f ./DatasetPreload ]; then
	echo "Files not found, downloading."
	rm ./modelBest-Current
	rm ./DatasetPreload	
	wget 'https://www.dropbox.com/s/a9r81uo1b2jqoo7/hw2.zip?dl=1' -O hw2.zip
	unzip hw2.zip
	#rm hw2.zip
fi

python3 testModel.py $1 $2