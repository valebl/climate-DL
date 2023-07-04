#!/bin/bash
#SBATCH -A ict23_esp_C
#SBATCH -p m100_usr_prod
#SBATCH --time 05:00:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=0
#SBATCH --ntasks-per-node=128   # 8 tasks out of 128
# --gres=gpu:1          # 1 gpus per node out of 4
#SBATCH --job-name=prep-north
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=vblasone@ictp.it
#SBATCH -o /m100_work/ICT23_ESP_C/vblasone/climate-DL/preprocessing/prep-graph.out
#SBATCH -e /m100_work/ICT23_ESP_C/vblasone/climate-DL/preprocessing/prep-graph.err

source /m100/home/userexternal/vblasone/.bashrc

conda activate /m100/home/userexternal/vblasone/geometric

cd /m100_work/ICT23_ESP_C/vblasone/climate-DL/preprocessing/

## fvg
#python3 preprocessing_graphs_and_targets.py --output_path='/m100_work/ICT23_ESP_C/vblasone/climate-DL/data_fvg_preprocessed/' --log_file='log.txt' --lon_min=12.75 --lon_max=14.00 --lat_min=45.50 --lat_max=46.75 --interval=0.25 --time_dim=140256 --suffix="" --make_plots

## north
python3 preprocessing_graphs_and_targets.py --input_path='/m100_work/ICT23_ESP_C/SHARED/' --output_path='/m100_work/ICT23_ESP_C/vblasone/climate-DL/data_north_preprocessed/' --log_file='log.txt' --lon_min=6.75 --lon_max=14.00 --lat_min=43.75 --lat_max=47.00 --interval=0.25 --time_dim=140256 --suffix="" --make_plots

