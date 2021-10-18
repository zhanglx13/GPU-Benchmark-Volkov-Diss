#! /usr/bin/env bash

set -e

################################################################################
## Visualize the occupancy variation during execution.
##
## The output of the kernel execution is
## warpid  smid  tStart  tEnd
## which is saved in `result_alpha_maxOcc_wave_maxLat_warps.txt`
## This script first locate the proper file according user input alpha and occ
## by:
##   1. list all result_* files and match alpha and occ.
##   2. locate the file with the largest wave value, which corresponds to
##      `ITER=32`
##   3. Get the sm id with the largest occupancy value among all SMs.
#       This requires running the binary.
##   4. Run the binary again with the sm id to write the occ result to
##      `occ_sm$smid.txt`.
##   5. Plot the figure
##   6. Copy the figure from server
##   7. Visualize
##
## To execute a command on the server, simply append the command after
## ssh lxzhang@server.cs.utexas.edu
## Note that to use variables in the command, the variable needs to be wrapped
## in 'single quotes' while the whole command needs to be wrapped in
## "double quotes".
##
## On the server, draw.gp and plot.sh are working together to produce the
## figure, which shows the occupancy during 1/3 of the total execution time.
## Cycles per Wave, maxLat, and average occupancy are printed as well.
##
## $1: alpha
## $2: occ
## $3: anything. This arg is used as a placeholder to enter the remote mode
################################################################################
ARCH=Kepler

if [[ $# -lt 2 ]];then
    echo "Usage: ./view.sh <alpha> <occ> [optional arg]"
    exit 0
fi

ofilename="${ARCH}_alpha=$1_occ=$2.pdf"

if [[ $# -eq 3 ]]; then
    echo "Generating figure on the remote"
    ## list all result file, match alpha and occ, get largest wave
    rest=$(ssh lxzhang@aida.cs.utexas.edu "cd localwd/'$ARCH'_result; ls result_'$1'_*" \
               | awk -F_ -v occ=$2 '$3 == occ')

    ## Check if occ is valid
    if [[ "$rest"  == "" ]];
    then
        echo "   result_$1_$2_*.txt does not exist"
        exit 0
    else
        rest=$(ssh lxzhang@aida.cs.utexas.edu "cd localwd/'$ARCH'_result; ls result_'$1'_'$2'_*" \
                   | awk -F_ -v max=0 '{if($4>max){wave=$4;maxLat=$5;warps=$6;max=$4}}END{print wave,maxLat,warps}')
    fi

    ## extract wave, maxLat, warps from the filename
    wave=$(echo $rest | awk '{print $1}')
    maxLat=$(echo $rest | awk '{print $2}')
    warpsExt=$(echo $rest | awk '{print $3}')
    warps=${warpsExt%.txt}

    targetF="result_$1_$2_${wave}_${maxLat}_${warps}.txt"

    ## Get the smid with largest occ among all SMs
    ssh lxzhang@aida.cs.utexas.edu "cd localwd && ./genOcc '$ARCH'_result/'$targetF' '$warps'" > occSum.txt
    smid=$(sed -n '3,16p' occSum.txt | awk -v max=0 '{if($3>max){want=$1;max=$3}}END{print want}')
    cyclesPerWave=$(echo "$maxLat / $wave" | bc)

    ## Write the occ result to file occ_sm$smid.txt and Draw the figure
    ssh lxzhang@aida.cs.utexas.edu "cd localwd && ./genOcc '$ARCH'_result/'$targetF' '$warps' '$smid' && ./plot.sh '$smid' '$cyclesPerWave' '$ofilename' '$2' '$maxLat'" > occSum.txt
    ## Copy the figure from server to local
    scp lxzhang@aida.cs.utexas.edu:/u/lxzhang/localwd/figures/$ofilename ./figures/$ofilename
fi

if [[ -f ./figures/$ofilename ]]; then
    ## Visualize the figure
    if [[ "$OSTYPE" == "darwin"* ]]; then
	    open ./figures/$ofilename
    else
	    okular ./figures/$ofilename
    fi
else
    echo "$ofilename does not exist!"
fi
