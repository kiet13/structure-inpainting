#!/bin/bash

# input output smoothing_method
# $1      $2    $3              

if [[ $3 == 'L0' ]]
then

    # Install octave
    apt install octave
    apt install liboctave-dev

    # Load image package
    cd smoothing/L0
    octave --eval "pkg install -forge image"
    echo "pkg load image;" >> .octaverc

    python run_L0_octave.py --input_path $1 --output_path $2 --Lambda 0.04
fi

if [[ $3 == 'SGF' ]]
then
    cd smoothing/SGF
    cmake .
    make

    for imageName in $1/*
    do
        outputName="${imageName##*/}"
        outputName="$2/$outputName"
        ./SGF $imageName $imageName $outputName 8 0.05 0.1 3
    done
fi

cd ../..