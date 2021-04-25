#!/bin/bash
echo "Generating DCGAN images."
python3 testDCGAN.py
echo "Generating WGAN images."
python3 testWGAN.py
echo "Generating ACGAN images."
python3 testACGAN.py

echo "All image generation finished."
