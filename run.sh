#!/bin/sh
pip install -r requirements.txt
python3 seam.py --help

python seam.py test/1.jpg 213 240 0 test/1_e0.jpg
python seam.py test/1.jpg 213 240 1 test/1_e1.jpg
python seam.py test/1.jpg 213 240 2 test/1_e2.jpg
python seam.py test/1.jpg 213 240 666 test/1_e666.jpg 10

python seam.py test/2.jpg 213 212 0 test/2_e0.jpg
python seam.py test/2.jpg 213 212 1 test/2_e1.jpg
python seam.py test/2.jpg 213 212 2 test/2_e2.jpg
python seam.py test/2.jpg 213 212 666 test/2_e666.jpg 23

python seam.py test/3.jpg 199 168 0 test/3_e0.jpg
python seam.py test/3.jpg 199 168 1 test/3_e1.jpg
python seam.py test/3.jpg 199 168 2 test/3_e2.jpg
python seam.py test/3.jpg 199 168 666 test/3_e666.jpg 14

python seam.py test/14.jpg 200 168 0 test/14_e0.jpg
python seam.py test/14.jpg 200 168 1 test/14_e1.jpg
python seam.py test/14.jpg 200 168 2 test/14_e2.jpg
python seam.py test/14.jpg 200 168 666 test/14_e666.jpg 16

python seam.py test/x1.jpg 384 240 0 test/x1_e0.jpg
python seam.py test/x1.jpg 384 240 1 test/x1_e1.jpg
python seam.py test/x1.jpg 384 240 2 test/x1_e2.jpg
python seam.py test/x1.jpg 384 240 666 test/x1_e666.jpg 16

python seam.py test/x2.jpg 298 240 0 test/x1_e0.jpg
python seam.py test/x2.jpg 298 240 1 test/x1_e1.jpg
python seam.py test/x2.jpg 298 240 2 test/x1_e2.jpg
python seam.py test/x2.jpg 298 240 666 test/x1_e666.jpg 25

python seam.py test/x9.jpg 192 240 0 test/x9_e0.jpg
python seam.py test/x9.jpg 192 240 1 test/x9_e1.jpg
python seam.py test/x9.jpg 192 240 2 test/x9_e2.jpg
python seam.py test/x9.jpg 192 240 666 test/x9_e666.jpg 23

python seam.py test/x13.jpg 384 234 0 test/x13_e0.jpg
python seam.py test/x13.jpg 384 234 1 test/x13_e1.jpg
python seam.py test/x13.jpg 384 234 2 test/x13_e2.jpg
python seam.py test/x13.jpg 384 234 666 test/x13_e666.jpg 30
