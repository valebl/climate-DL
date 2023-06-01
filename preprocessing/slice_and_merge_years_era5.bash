#!/bin/bash

cd '/m100_work/ICT23_ESP_C/vblasone/ERA5/'

for v in 'q' 't' 'u' ; do #'v' 'z' ; do
    files=$(ls ${v}_*.nc)
    for file in $files ; do
        cdo sellonlatbox,12.0,14.75,44.75,47.25 /m100_work/ICT23_ESP_C/vblasone/ERA5/${file} /m100_work/ICT23_ESP_C/vblasone/NORTH_ITALY/fvg/sliced_${file}
    done
done


cd '/m100_work/ICT23_ESP_C/vblasone/NORTH_ITALY/fvg/'

for v in 'q' 't' 'u' 'v' 'z' ; do
    cdo -O -f nc4 -z zip -L -b F32 mergetime sliced_${v}_2001.nc sliced_${v}_2002.nc sliced_${v}_2003.nc sliced_${v}_2004.nc sliced_${v}_2005.nc sliced_${v}_2006.nc sliced_${v}_2007.nc sliced_${v}_2008.nc sliced_${v}_2009.nc sliced_${v}_2010.nc sliced_${v}_2011.nc sliced_${v}_2012.nc sliced_${v}_2013.nc sliced_${v}_2014.nc sliced_${v}_2015.nc sliced_${v}_2016.nc sliced_${v}.nc
done

