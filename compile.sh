#!/bin/bash

nvcc -arch sm_20 gravity.cu -g -o bin/grav;
nvcc -arch sm_20 gravity-fast.cu -g -o bin/grav-fast;

