#!/bin/bash

if [ -z "$1" ]
then
   echo "Please specify a target path to this script, e.g.: /setup_mvsanywhere.sh /path/to/mvsanywhere";
   exit 1
fi

set -e
TARGET="$1"

echo "Downloading MVSAnywhere repository https://github.com/nianticlabs/mvsanywhere to $TARGET"
mkdir -p "$1"

git clone https://github.com/nianticlabs/mvsanywhere "$TARGET"

OLD_PWD="$PWD"
cd $TARGET

mkdir pretrained_weights
wget https://storage.googleapis.com/niantic-lon-static/research/mvsanywhere/mvsanywhere_hero.ckpt -O $TARGET/pretrained_weights/mvsanywhere_hero.ckpt
wget https://storage.googleapis.com/niantic-lon-static/research/mvsanywhere/mvsanywhere_dot.ckpt -O $TARGET/pretrained_weights/mvsanywhere_dot.ckpt

cd "$OLD_PWD"
echo "Done"
exit 0
