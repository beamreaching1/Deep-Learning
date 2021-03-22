To run the trained model on a dataset four files are needed.
Two of the files come included in the GitHub repository (testModel.py and hw2_seq2seq.sh).
The other two files (modelBest and DatasetPreload) will be downloaded by the shell script hw2_seq2seq.sh when it is run.

The syntax for running the shell script is ./ hw2_seq2seq.sh dataDir results.txt

dataDir must be the path to the test root folder of the dataset and nothing more. It also is expected to have the same file structure as MLDS including a named label file called “testing_label.json”. 
results.txt can be any name but the .txt file extension is not added by default.

This program was tested on a Palmetto Cluster node but should work on any Linux system that is correctly setup. 
The program also requires that a compatible GPU be present (one with at least 6GB of memory and compute capability 6.0 or higher). For my test I used a Tesla V100.

Palmetto Node Launch Parameters: 
qsub -I -l select=1:ncpus=8:mpiprocs=8:ngpus=1:gpu_model=v100:mem=62gb:interconnect=hdr,walltime=4:00:00
