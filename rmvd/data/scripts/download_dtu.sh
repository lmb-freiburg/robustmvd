#!/bin/bash

if [ -z "$1" ]
then
   echo "Please pass a target path to this script, e.g.: /download_dtu.sh /path/to/dtu.";
   exit 1
fi

TARGET="$1"

echo "Downloading DTU dataset to $TARGET."
mkdir -p "$1"

OLD_PWD="$PWD"
cd $TARGET

# All steps are skipped if their output is already present, so the script can
# safely be executed again if a later step failed.

# 1. Download the rectified images from the original dataset website:
# http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip and extract it.
if [ -d "Rectified" ]; then
  echo "Step 1: 'Rectified' already exists, skipping download."
else
  wget --no-check-certificate "http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip"
  unzip Rectified.zip
  rm Rectified.zip
fi

# 2. Download the pointclouds from the original dataset website:
# http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip and extract it.
if [ -d "Points" ]; then
  echo "Step 2: 'Points' already exists, skipping download."
else
  wget --no-check-certificate "http://roboimagedata2.compute.dtu.dk/data/MVS/Points.zip"
  unzip Points.zip
  rm Points.zip
fi

# Original GT depths download step, which is outdated now because download link is not valid anymore:
# Download the file `dtu.tar.xz` from https://polybox.ethz.ch/index.php/s/ugDdJQIuZTk4S35 (supplied by the
# PatchmatchNet repository https://github.com/FangjinhuaWang/PatchmatchNet) and extract it.
# wget --no-check-certificate "https://polybox.ethz.ch/index.php/s/ugDdJQIuZTk4S35/download" -O dtu.tar.xz
# tar xf dtu.tar.xz
# rm dtu.tar.xz

# 3. Download GT depths from Google Drive, as provided by https://github.com/YoYo000/MVSNet/issues/106.
# Note that Google Drive often blocks gdown downloads for this file.
if [ -d "Depths" ]; then
  echo "Step 3: 'Depths' already exists, skipping download."
else
  gdown https://drive.google.com/open?id=1LVy8tsWajG3uPTCYPSxDvVXFCdIYXaS-

  if [ ! -f "Depths.zip" ]; then
    echo ""
    echo "ERROR: Could not download 'Depths.zip'."
    echo "Google Drive frequently blocks gdown for this file. You can either:"
    echo "  a) wait a while and execute this script again, or"
    echo "  b) download the file manually in a browser from"
    echo "     https://drive.google.com/open?id=1LVy8tsWajG3uPTCYPSxDvVXFCdIYXaS-"
    echo "     move it to $TARGET/Depths.zip and extract it there;"
    echo "     it must result in $TARGET/Depths"
    cd "$OLD_PWD"
    exit 1
  fi

  unzip Depths.zip
  rm Depths.zip
fi

cd "$OLD_PWD"
echo "Done"
exit 0
