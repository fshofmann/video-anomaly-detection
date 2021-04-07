#!/bin/bash

# To be executed within the upCam data repository folder. Directory structure must be already preprocessed.
# Extracts all raw videos and compresses them to a smaller res., before putting results into a new folder next to raw.
# Also changes the frame rate to a fixed instead of a variable one.

width=128
height=64

for folder in $(seq -f "%02g" 0 9); do
  mkdir -p preprocessed_$width*$height/"$folder"
  for i in raw/"$folder"/*.mp4; do
    # Keep old (potentially variable) frame rate
    # ffmpeg -i "$i" -vsync vfr -vf scale=$width:$height "${i/raw/preprocessed_$width*$height}"
    # Use constant (5) frame rate
    ffmpeg -i "$i" -r 5 -vf scale=$width:$height "${i/raw/preprocessed_$width*$height}"
  done
done
