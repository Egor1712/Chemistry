#!/usr/bin/gnuplot -persist
#reset
set datafile separator ','

#fname  = 'L32_D0.2000_V0.3570_tb0.8000_tp0.5000_tn0.0000_tpn0.0000_J0.2500_results.csv'
#labeltext = 'this is label'


zerolevel = 0.1

set xrange[0:0.8] 
set yrange[0:0.22]

set label labeltext at 0.01, 0.18 

set xyplane relative 1.5

set xlabel "n"
set ylabel "T"
set zlabel "O.P."

set hidden3d


#set dgrid3d 70,70,box 0.05 -- завышает t перехода
#set dgrid3d 70,70, gauss 0.005 -- завышает t перехода
#set dgrid3d 70,70, cauchy 0.0001 -- стращный, по прежнему завышает t
#set dgrid3d 70,70, exp 0.01 -- OK
#set dgrid3d 70,70, hann 0.05 -- разрывы

set dgrid3d 70,70, gauss 0.1, 0.005 #-- максимально хорош для хим потенциала, но по разному сглаживает x и y 

set contour
set cntrlabel onecolor 
set cntrparam levels discrete zerolevel
set isosamples 40

set view map
unset surface

set table fname.'AFM_contour.dat'
splot fname u 3:2:10  w l lc 'red' lw 4 ti 'AFM'

set table fname.'CO_contour.dat'
splot fname u 3:2:6  w l lc 'blue' lw 4 ti 'CO'

set table fname.'SC_contour.dat'
splot fname u 3:2:13  w l lc 'green' lw 4 ti 'SC'

set table fname.'FL_contour.dat'
splot fname u 3:2:12  w l lc 'violet' lw 4 ti 'FL'
