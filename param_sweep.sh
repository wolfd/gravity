#!/bin/bash

./compile.sh;
mkdir -p sweep;
mkdir -p sweep/inputs;
mkdir -p sweep/saved;
mkdir -p sweep/outputs;
for v in $(./linspace.sh 0.5 4.0 100);
do
        echo $m;
        ./explode.py -m $m -n 994;
        cp input.csv sweep/inputs/$m-input.csv;
        ./bin/grav;
        mv output.csv sweep/outputs/$m-output.csv;
        mv next-input.csv sweep/saved/$m-next-input.csv;
done

echo "Done!";

