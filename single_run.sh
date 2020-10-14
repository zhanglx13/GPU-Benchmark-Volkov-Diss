#! /usr/bin/env bash

## $1: ITER
## $2: ALPHA
## $3: BS
## $4: SMEM
##
## The above 4 args are used to compile main.cu. The following arg is used to
## print the maxOcc info. 
## 
## $5: maxOcc


if [[ $# -lt 4 ]];then
    echo "Need 4 args: ./single_run.sh <ITER> <ALPHA> <BS> <SMEM>"
    exit 0
fi
#echo ">>>>>>>>>> ALPHA=$2 maxOcc=$5 ITER=$1 <<<<<<<<<<"
## Compile
nvcc -arch=sm_35 -DITER=$1 -DALPHA=$2 -DBS=$3 -DSMEM=$4 -DEXPERIMENT main.cu -o main -Xptxas -v &> compile_dump
#tail -n 1 compile_dump

limit=10
occThrRatio=0.95
flag=1

while [[ "$flag" -eq "1" ]]; do

    cnt=0

    ./main 1 > occSum.txt

    maxOcc=$(sed -n '4p' occSum.txt | awk '{print $2}')
    occThr=$(echo "$maxOcc * $occThrRatio" | bc)
    aveOcc=$(grep "Ave Occupancy:" occSum.txt | awk '{print $3}')
    #head -n 1 occSum.txt
    warps=$(head -n 1 occSum.txt | awk '{print $6}')
    warps=${warps%,}
    echo "Max occ: $maxOcc   target occ: $occThr   warps: $warps"
    echo "$cnt: aveOcc = $aveOcc"
    cnt=$((cnt + 1))


    while [[ $(echo "$aveOcc < $occThr" | bc) -eq 1 ]] && [[ "$cnt" -lt "$limit" ]]; do
        ./main 1 > occSum.txt
        if [[ $? -eq 139 ]]; then
            echo "$cnt: seg fault"
        else
            aveOcc=$(grep "Ave Occupancy:" occSum.txt | awk '{print $3}')
            echo "$cnt: aveOcc = $aveOcc"
            cnt=$((cnt + 1))
        fi 
    done

    if [[ $(echo "$aveOcc > $occThr" | bc) -eq 1 ]]; then
        ## Obtain file name from the output
        wave=$(grep "Ave Waves:" occSum.txt | awk '{print $3}')
        maxLat=$(grep "Max Latency:" occSum.txt | awk '{print $3}')
        arithThr=$(grep "Ave throughput:" occSum.txt | awk '{print $3}')
        echo "$2   $maxOcc   $arithThr   $1   $3   $4   $maxLat   $warps" >>  arithThr_$2.txt
        fileToCopy=result_$2_${maxOcc}_${wave}_${maxLat}_*.txt
        echo "Copying $fileToCopy to server ..."
        ./toServer.sh $fileToCopy
        ## Remove the files in case dropbox space explodes
        rm result_$2_${maxOcc}_*.txt
        flag=0
    else
        echo "Decrease threshold and try again"
        occThrRatio=$(echo "$occThrRatio - 0.01" | bc)
    fi

done



