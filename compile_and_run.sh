#!/bin/bash

nvcc -arch sm_20 gravity.cu -o bin/grav &&
    ./bin/grav &&
    ./render.py;

