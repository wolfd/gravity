#!/bin/bash

nvcc -arch sm_20 gravity.cu -g -o bin/grav &&
    ./bin/grav &&
    ./render.py -n 606;

