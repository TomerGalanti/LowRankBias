#!/bin/bash

read -p "Please list xids: " xid_list

IFS=", " read -r -a xids <<< "$xid_list"

for xid in "${xids[@]}"; do
    sbatch --export=ALL,xid=$xid --job-name=${xid}_train retrain.sh
done
