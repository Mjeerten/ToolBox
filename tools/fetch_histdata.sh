#!/bin/bash
# HistData 1-minute ASCII: whole years for past years, months for the current year.
# Usage: bash fetch_histdata.sh [SYMBOL] [DIR]   (default NSXUSD into hd/)
sym=${1:-NSXUSD}; dir=${2:-hd}; low=$(echo "$sym" | tr 'A-Z' 'a-z')
get() { # $1 = page suffix, $2 = date, $3 = datemonth, $4 = out
  [[ -s $4 ]] && return
  page="https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/$low/$1"
  tk=$(curl -sS -A "Mozilla/5.0" -c ck.txt "$page" | grep -o 'id="tk" value="[^"]*"' | head -1 | sed 's/.*value="//;s/"//')
  curl -sS -A "Mozilla/5.0" -b ck.txt -e "$page" -X POST https://www.histdata.com/get.php \
    -d "tk=$tk&date=$2&datemonth=$3&platform=ASCII&timeframe=M1&fxpair=$sym" -o $4
  echo "$4 $(stat -c %s $4)"; sleep 3
}
mkdir -p $dir
for y in 2019 2020 2021 2022 2023 2024 2025; do get $y $y $y $dir/$y.zip; done
for m in 1 2 3 4 5 6 7 8 9; do get 2026/$m 2026 2026$(printf %02d $m) $dir/2026_$m.zip; done
