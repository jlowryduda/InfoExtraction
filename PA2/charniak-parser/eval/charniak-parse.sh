#!/bin/sh

WSJDIR=/home/j/clp/chinese/corpora/wsj

#train
#../TRAIN/allScript En ../DATA/EN/ "$WSJDIR/00/*.mrg" "$WSJDIR/01/wsj_0101.mrg"

#parse
../PARSE/parseIt ../DATA/EN/ ../sample-sentences
