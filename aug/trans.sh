#!/bin/bash

set -e

for i in ./*idx3-ubyte
do
    echo $i
    python -u trans.py --file $i
done
