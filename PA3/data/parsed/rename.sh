#!/bin/bash
for file in *.parsed
do
    mv $file ${file/.tagged/}
done
