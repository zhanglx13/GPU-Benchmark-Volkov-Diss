#! /usr/bin/env bash

##
## Process Kepler_CC35_arithThr_ALPHA.txt to compute the average throughput
## of the same maxOcc. The result file is named as Kepler_CC35_aveArithThr_ALPHA.txt
##

## $1: alpha
echo "ALPHA  occ      thr    BS   SMEM   maxLat" > Kepler_CC35_aveArithThr_$1.txt 

#maxOcc=2
for maxOcc in `seq 1 64`
do
    awk -v occ="^$maxOcc$" '$2 ~ occ' Kepler_CC35_arithThr_$1.txt > tmp_$maxOcc
    if [[ -s tmp_$maxOcc ]]; then
        aveThr=$(awk -v N=3 '{ sum += $N } END { if (NR > 0) print sum / NR }' tmp_$maxOcc)
        aveBS=$(awk -v N=5 '{ sum += $N } END { if (NR > 0) print sum / NR }' tmp_$maxOcc)
        aveSMEM=$(awk -v N=6 '{ sum += $N } END { if (NR > 0) print sum / NR }' tmp_$maxOcc)
        aveMaxLat=$(awk -v N=7 '{ sum += $N } END { if (NR > 0) printf "%d", sum / NR }' tmp_$maxOcc)
        printf "%5d  %3d  %f  %3d  %5d   %d\n" $1 $maxOcc $aveThr $aveBS $aveSMEM $aveMaxLat | tee -a Kepler_CC35_aveArithThr_$1.txt
    else
        echo "empty file!!"
    fi
    rm tmp_$maxOcc
done
