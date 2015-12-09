#!/bin/bash

./compile.sh;
mkdir -p sweep;
mkdir -p sweep/inputs;
mkdir -p sweep/saved;
mkdir -p sweep/outputs;
for v in $(./linspace.sh 6 20 100);
do
        echo $v;
        ./explode.py -v $v -n 994;
        cp input.csv sweep/inputs/$v-input.csv;
        ./bin/grav;
        mv output.csv sweep/outputs/$v-output.csv;
        mv next-input.csv sweep/saved/$v-next-input.csv;
done

echo "Done!";

