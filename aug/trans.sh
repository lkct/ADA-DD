#!/bin/bash

set -e

ls *idx3-ubyte | xargs -n 1 -I {} -P 10 sh -c 'echo {};python -u trans.py --file {}'
