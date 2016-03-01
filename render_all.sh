#!/bin/bash

mkdir -p renders;


# make sure you run the other one first and it's done with something.
# yeah it's not great.
for m in $(./linspace.sh 0.5 4.0 10);
do
		#scp -rp dwolf@deepthought:~/wolf/gravity/sweep/saved sweep_inputs
        echo $m;
        ./render.py -n 1000 -f "second-outputs/2nd-output-$m.csv" -i "renders/second-orbit-$m" &
done

echo "Done!";

