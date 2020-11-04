#! /usr/bin/env bash

## Batch run experiments for all occupancy (1-64) for a single ALPHA

## $1: ALPHA

## Include occ config script containing BS and SMEM set up for each occ value
source occConfig.sh

if [[ $# -eq 0 ]]; then
    echo "Need to specify alpha: ./batch_run.sh <ALPHA>"
    exit 0
fi

echo "ALPHA  maxOcc   arithThr   ITER   BS   SMEM  maxLat   warps" > arithThr_$1.txt
for occ in `seq 64 -1 1`
do
    if [[ ${bs[$occ]} ]]; then
        echo "###########################################"
        printf "##      ALPHA = %3d     maxOcc = %2d      ##\n" $1 $occ
        #echo "## ALPHA=$1 maxOcc=$occ ##"
        echo "###########################################"
        for iter in 32 64 96 128 160 192 224 256 288 320
        do
            echo ""
            echo ">>> ITER = $iter  ALPHA = $1  maxOcc = $occ"
            echo ""
            ./single_run.sh $iter $1 ${bs[$occ]} ${smem[$occ]}
        done
    fi
done
