#!/bin/bash
# Download Pascal VOC 2012 trainval (official VOCtrainval_11-May-2012.tar, 1,999,639,040 bytes,
# md5 6cd6e144f989b92b3379bac3b3de84fd) and extract it in the official layout.
# Usage: bash download_voc2012.sh <DATA_ROOT>   -> <DATA_ROOT>/VOCdevkit/VOC2012 (set VOC2012_DIR in ../paths.py to it)
# Mirrors are tried in order; curl resumes partial downloads. Idempotent.
set -u
DST=${1:?usage: bash download_voc2012.sh <DATA_ROOT>}
TAR=$DST/VOCtrainval_11-May-2012.tar; OUT=$DST/VOCdevkit/VOC2012
MD5=6cd6e144f989b92b3379bac3b3de84fd; SIZE=1999639040
URLS="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar https://data.pjreddie.com/files/VOCtrainval_11-May-2012.tar"
mkdir -p "$DST"
if [ -f "$OUT/VOC2012_OK" ]; then echo "already done: $OUT"; exit 0; fi
if [ ! -f "$TAR" ] || [ "$(stat -c %s "$TAR")" -ne "$SIZE" ]; then
  for u in $URLS; do
    echo "downloading $u"
    curl -L -C - --retry 5 --retry-delay 20 -o "$TAR" "$u" && [ "$(stat -c %s "$TAR")" -eq "$SIZE" ] && break
    echo "  incomplete from $u ($(stat -c %s "$TAR" 2>/dev/null || echo 0) bytes); trying the next mirror"
  done
fi
[ "$(stat -c %s "$TAR")" -eq "$SIZE" ] || { echo "download failed (size $(stat -c %s "$TAR"))"; exit 3; }
echo "md5 check"; have=$(md5sum "$TAR" | cut -d' ' -f1)
[ "$have" = "$MD5" ] || { echo "MD5 MISMATCH: $have != $MD5"; exit 4; }
echo "extracting"; tar -xf "$TAR" -C "$DST" || { echo "tar failed"; exit 5; }
ok=1
for p in JPEGImages:17125 SegmentationClass:2913; do
  d=${p%%:*}; want=${p##*:}; have=$(ls "$OUT/$d" | wc -l); echo "  $d: $have (want $want)"; [ "$have" -eq "$want" ] || ok=0
done
for s in train:1464 val:1449; do
  f=${s%%:*}; want=${s##*:}; have=$(grep -c . "$OUT/ImageSets/Segmentation/$f.txt"); echo "  Segmentation/$f.txt: $have (want $want)"; [ "$have" -eq "$want" ] || ok=0
done
if [ $ok -eq 1 ]; then date > "$OUT/VOC2012_OK"; echo "VOC 2012 ready: $OUT"; else echo "COUNT MISMATCH"; exit 6; fi
