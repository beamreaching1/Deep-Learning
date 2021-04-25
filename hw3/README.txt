To run this program either type ./run.sh or python3 testXGAN.py where X is AC, DC, or W.
To test the FID score of each of these models use...

python3 -m pytorch_fid --device cuda:0 ./fakeX/ ./realX/

where X is AC, DC, or W.

These models were run and tested on the Clemson Palmetto Cluster using this command:
qsub -I -l select=1:ncpus=8:mpiprocs=8:ngpus=1:gpu_model=v100s:mem=64gb:interconnect=hdr,walltime=0:10:00