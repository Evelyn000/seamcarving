#!/bin/sh
pip install -r requirements.txt
python3 seam.py 

python seam.py 1.jpg 213 240 0 1_e0.jpg
python seam.py 1.jpg 213 240 1 1_e1.jpg
python seam.py 1.jpg 213 240 2 1_e2.jpg
python seam.py 1.jpg 213 240 666 1_e666.jpg layer10

python seam.py 2.jpg 213 212 0 2_e0.jpg
python seam.py 2.jpg 213 212 1 2_e1.jpg
python seam.py 2.jpg 213 212 2 2_e2.jpg
python seam.py 2.jpg 213 212 666 2_e666.jpg layer23

python seam.py 3.jpg 199 168 0 3_e0.jpg
python seam.py 3.jpg 199 168 1 3_e1.jpg
python seam.py 3.jpg 199 168 2 3_e2.jpg
python seam.py 3.jpg 199 168 666 3_e666.jpg layer14

python seam.py 14.jpg 200 168 0 14_e0.jpg
python seam.py 14.jpg 200 168 1 14_e1.jpg
python seam.py 14.jpg 200 168 2 14_e2.jpg
python seam.py 14.jpg 200 168 666 14_e666.jpg layer16