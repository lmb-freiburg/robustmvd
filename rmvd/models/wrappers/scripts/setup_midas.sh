#!/bin/bash

if [ -z "$1" ]
then
   echo "Please specify a target path to this script, e.g.: /setup_midas.sh /path/to/midas";
   exit 1
fi

set -e
TARGET="$1"

echo "Downloading MiDaS repository https://github.com/isl-org/MiDaS to $TARGET"
mkdir -p "$1"

git clone https://github.com/isl-org/MiDaS "$TARGET"

OLD_PWD="$PWD"
cd $TARGET

cd weights
wget --no-check-certificate https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt
wget --no-check-certificate https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt

cd "$OLD_PWD"
echo "Done"
exit 0
