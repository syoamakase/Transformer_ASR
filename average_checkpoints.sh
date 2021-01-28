#!/bin/bash

load_dir=${1}
start=${2:-141}
end=${3:-150}

python average_checkpoints.py --backend pytorch --snapshots ${load_dir}/network.epoch* --out ${load_dir}/network.average_epoch${start}-epoch${end} --start ${start} --end ${end}
