#!/usr/bin/bash
#search_dir=$1
#search_dir='d03tb9tp8v-variation'
#for entry in "$search_dir"/*.csv
for entry in *.csv
do
  fname=$(realpath "$entry")
  bname=$(basename "$entry" .csv)
#  ofname="$search_dir/$bname.png"
  ofname="data/$bname.png"
  gnuplot <<- EOF
        reset
        fname = '$fname'
        labeltext = '$entry'
        set terminal pngcairo size 800, 600 fontscale 1
        set output "$ofname"
        load 'next_region.plt'
        
EOF
done
