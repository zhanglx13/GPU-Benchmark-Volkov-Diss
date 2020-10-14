#! /usr/bin/env bash

## $1: smid

scp lxzhang@aida.cs.utexas.edu:/u/lxzhang/localwd/occ_sm$1.pdf ./

if [[ "$OSTYPE" == "darwin"* ]]; then
	open occ_sm$1.pdf
else
	okular occ_sm$1.pdf
fi	
