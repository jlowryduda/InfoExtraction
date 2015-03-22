#!/bin/bash
for file in *.parsed
do
    echo mv $file ${file/.tagged/}
done
