#!/bin/bash

./compile.sh;
mkdir -p sweep;
mkdir -p sweep/inputs;
mkdir -p sweep/saved;
mkdir -p sweep/outputs;
for m in $(./linspace.sh 0.5 4.0 100);
do
        echo $m;
        ./explode.py -m $m;
        cp input.csv "sweep/inputs/input-$m.csv";
        ./bin/grav;
        mv output.csv "sweep/outputs/output-$m.csv";
        mv next-input.csv "sweep/saved/next-input-$m.csv";
done

echo "Done!";

