#!/bin/bash
# Download and extract ADE20K SceneParse150 (ADEChallengeData2016.zip) and verify it.
# Usage: bash prepare_ade20k.sh <DATA_ROOT>     -> <DATA_ROOT>/ADEChallengeData2016 (set ADE20K_DIR in ../paths.py to it)
# Idempotent: skips the download if the zip is present, `unzip -n` keeps existing files, and every extracted image /
# annotation is size-checked against the zip listing (a truncated file from an interrupted extraction is re-extracted).
set -u
DST=${1:?usage: bash prepare_ade20k.sh <DATA_ROOT>}
URL=http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
ZIP=$DST/ADEChallengeData2016.zip
OUT=$DST/ADEChallengeData2016
mkdir -p "$DST"
if [ ! -f "$ZIP" ]; then
  echo "downloading $URL"
  curl -L -C - --retry 5 -o "$ZIP.part" "$URL" && mv "$ZIP.part" "$ZIP" || { echo "download failed"; exit 2; }
fi
echo "unzip -n $ZIP -> $DST"
unzip -n -q "$ZIP" -d "$DST" || { echo "unzip failed"; exit 3; }
echo "size check against the zip listing"
unzip -l "$ZIP" | awk 'NR>3 && $1 ~ /^[0-9]+$/ && $4 ~ /\.(jpg|png)$/ {print $1, $4}' | while read sz path; do
  have=$(stat -c %s "$DST/$path" 2>/dev/null || echo -1)
  [ "$have" -eq "$sz" ] || { echo "  re-extracting $path (zip $sz, disk $have)"; unzip -o -q "$ZIP" "$path" -d "$DST"; }
done
ok=1
for p in images/training:20210 images/validation:2000 annotations/training:20210 annotations/validation:2000; do
  d=${p%%:*}; want=${p##*:}; have=$(ls "$OUT/$d" | wc -l)
  echo "  $d: $have (want $want)"; [ "$have" -eq "$want" ] || ok=0
done
if [ $ok -eq 1 ]; then date > "$OUT/ADE20K_OK"; echo "ADE20K ready: $OUT"; else echo "COUNT MISMATCH - rerun"; exit 4; fi
