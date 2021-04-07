#!/bin/bash

# Bash script version of the frame splitter. Used for documentation purposes only.

mkdir -p frames
ffmpeg -i FILENAME.mp4 frames/%06d.png
