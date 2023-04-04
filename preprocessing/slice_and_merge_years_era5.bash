#!/bin/bash

cd '/m100_work/ICT23_ESP_C/vblasone/ERA5/'

for v in 'q' 't' 'u' ; do #'v' 'z' ; do
    files=$(ls ${v}_*.nc)
    for file in $files ; do
        cdo sellonlatbox,12.0,14.75,44.75,48.0 /m100_work/ICT23_ESP_C/vblasone/ERA5/${file} /m100_work/ICT23_ESP_C/vblasone/NORTH_ITALY/${file}_sliced.nc
    done
done


cd '/m100_work/ICT23_ESP_C/vblasone/NORTH_ITALY/'

for v in 'q' 't' 'u' 'v' 'z' ; do
    cdo -O -f nc4 -z zip -L -b F32 mergetime ${v}_2001.nc_sliced.nc ${v}_2002.nc_sliced.nc ${v}_2003.nc_sliced.nc ${v}_2004.nc_sliced.nc ${v}_2005.nc_sliced.nc ${v}_2006.nc_sliced.nc ${v}_2007.nc_sliced.nc ${v}_2008.nc_sliced.nc ${v}_2009.nc_sliced.nc ${v}_2010.nc_sliced.nc ${v}_2011.nc_sliced.nc ${v}_2012.nc_sliced.nc ${v}_2013.nc_sliced.nc ${v}_2014.nc_sliced.nc ${v}_2015.nc_sliced.nc ${v}_2016.nc_sliced.nc ${v}_sliced_fvg.nc
done
