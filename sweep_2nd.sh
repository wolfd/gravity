#!/bin/bash

mkdir -p sweep_inputs;
mkdir -p sweep/inputs;
mkdir -p sweep/saved;
mkdir -p sweep/outputs;


# make sure you run the other one first and it's done with something.
# yeah it's not great.
for m in $(./linspace.sh 0.5 4.0 10);
do
		scp -rp dwolf@deepthought:/data1/wolf/gravity/sweep/saved sweep_inputs
        echo $m;
        cp "sweep_inputs/input-$m.csv" input.csv;
        ./bin/grav-fast;
        mv output.csv "sweep/outputs/2nd-output-$m.csv";
        mv next-input.csv "sweep/saved/2nd-next-input-$m.csv";
done

echo "Done!";

