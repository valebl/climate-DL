#!/bin/bash 
#SBATCH -A ict23_esp_0
#SBATCH --partition=boost_usr_prod
#SBATCH --time 00:30:00       # format: HH:MM:SS
#SBATCH -N 1                  # 1 node
#SBATCH --mem=60G
## SBATCH --ntasks-per-node=32   # 8 tasks out of 128
#SBATCH --gres=gpu:1          # 1 gpus per node out of 4
#SBATCH --job-name=ann
## SBATCH --mail-type=FAIL,END
## SBATCH --mail-user=vblasone@ictp.it
#SBATCH -o /leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/extreme_2002_ERA5/large/run.out
#SBATCH -e /leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/extreme_2002_ERA5/large/run.err

source /leonardo/home/userexternal/vblasone/.bashrc

module purge
module load --auto profile/deeplrn
module load gcc
module load cuda/11.8

conda activate /leonardo_work/ICT23_ESP_0/env/GNNenv

cd /leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/

#year=2008
#year_txt="2008"

#"/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/data_fvg_preprocessed/"

python main.py --input_path="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/north_italy/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/extreme_2002_ERA5/large/" --test_graph_file="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/cl_north_italy/checkpoint_4.pth" --checkpoint_reg="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_north_italy/checkpoint_29.pth" --input_file="input_standard.pkl" --model_name_cl="Classifier_edges_test_large" --model_name_reg="Regressor_edges_test_large" --output_file="G_predictions.pkl" --log_file="log.txt" --make_plots --img_extension="jpg" --year_start=2002 --month_start=11 --day_start=22 --year_end=2002 --month_end=11 --day_end=30 --first_year=2001 --cmap='jet' --lon_min=6.75 --lon_max=14.00 --lat_min=43.75 --lat_max=47.00 --large_graph

# north italy
#python main.py --input_path="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/north_italy/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/north_italy_ann/" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/test_cl_ann/checkpoint_5.pth" --checkpoint_reg="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_north_italy/checkpoint_29.pth" --input_file="input_standard.pkl" --lat_dim=15 --lon_dim=31 --model_name_cl="Classifier_ann_test" --model_name_reg="Regressor_edges_test" --output_file="G_predictions.pkl" --log_file="log.txt" --make_plots --img_extension="jpg" --year_start=2016 --month_start=1 --day_start=1 --year_end=2016 --month_end=12 --day_end=31 --first_year=2001 --cmap='jet'

# sicilia from north italy
#python main.py --input_path="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/sicilia/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/sicilia/" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/cl_north_italy/checkpoint_4.pth" --checkpoint_reg="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_north_italy/checkpoint_29.pth" --input_file="input_standard.pkl" --lon_dim=15 --lat_dim=12 --model_name_cl="Classifier_edges_test" --model_name_reg="Regressor_edges_test" --output_file="G_predictions_test.pkl" --log_file="log.txt" --make_plots --img_extension="png" --year_start=2001 --month_start=1 --day_start=1 --year_end=2016 --month_end=12 --day_end=31 --first_year=2001 --cmap='turbo' --no-large_graph

# sardegna from north italy
#python main.py --input_path="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/sardegna/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/sardegna/" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/cl_north_italy/checkpoint_4.pth" --checkpoint_reg="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_north_italy/checkpoint_29.pth" --input_file="input_standard.pkl" --lon_dim=10 --lat_dim=16 --model_name_cl="Classifier_edges_test" --model_name_reg="Regressor_edges_test" --output_file="G_predictions_test.pkl" --log_file="log.txt" --make_plots --img_extension="png" --year_start=2001 --month_start=1 --day_start=1 --year_end=2016 --month_end=12 --day_end=31 --first_year=2001 --cmap='turbo' --no-large_graph 

# italy ALP-3
#python main.py --input_path="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/italy_alp3/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/alp3_${year}/" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/cl_north_italy/checkpoint_4.pth" --checkpoint_reg="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_north_italy/checkpoint_29.pth" --input_file="input_standard.pkl" --lat_dim=35 --lon_dim=44 --model_name_cl="Classifier_edges_test" --model_name_reg="Regressor_edges_test" --output_file="G_predictions_test.pkl" --log_file="log.txt" --make_plots --img_extension="png" --year_start=${year} --month_start=1 --day_start=1 --year_end=${year} --month_end=12 --day_end=31 --first_year=2000 --cmap='turbo' --no-large_graph

# italy
#python main.py --input_path="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/italy/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/italy_2/" --idx_file="idx_test.pkl" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/cl_north_italy/checkpoint_4.pth" --checkpoint_reg="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_north_italy/checkpoint_29.pth" --input_file="input_standard.pkl" --lat_dim=44 --lon_dim=49 --model_name_cl="Classifier_edges_test" --model_name_reg="Regressor_edges_test" --output_file="G_predictions_2016_test.pkl" --log_file="log.txt" --make_plots --img_extension="jpg"

# north italy
#python main.py --input_path="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/north_italy/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/north_italy/" --idx_file="idx_test.pkl" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/cl_north_italy/checkpoint_4.pth" --checkpoint_reg="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_north_italy/checkpoint_29.pth" --input_file="input_standard.pkl" --lat_dim=15 --lon_dim=31 --model_name_cl="Classifier_edges_test" --model_name_reg="Regressor_edges_test" --output_file="G_predictions_2016_test.pkl" --log_file="log.txt" --make_plots --img_extension="jpg"

# central from north italy
#python main.py --input_path="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/central_italy/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/central_from_north_italy_2001-2016/" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/cl_north_italy/checkpoint_4.pth" --checkpoint_reg="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_north_italy/checkpoint_29.pth" --input_file="input_standard.pkl" --lat_dim=13 --lon_dim=18 --model_name_cl="Classifier_edges_test" --model_name_reg="Regressor_edges_test" --output_file="G_predictions_test.pkl" --log_file="log.txt" --make_plots --img_extension="png" --year_start=2001 --month_start=1 --day_start=1 --year_end=2016 --month_end=12 --day_end=31 --cmap='turbo' --no-large_graph

# south from north italy 
#python main.py --input_path="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/south_italy/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/south_from_north_italy_2001-2016/" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/cl_north_italy/checkpoint_4.pth" --checkpoint_reg="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_north_italy/checkpoint_29.pth" --input_file="input_standard.pkl" --lat_dim=22 --lon_dim=20 --model_name_cl="Classifier_edges_test" --model_name_reg="Regressor_edges_test" --output_file="G_predictions_2016_test.pkl" --log_file="log.txt" --make_plots --img_extension="jpg" --year_start=2001 --month_start=1 --day_start=1 --year_end=2016 --month_end=12 --day_end=31 --first_year=2001 --cmap='turbo' --no-large_graph

# fvg from north italy
#python main.py --input_path="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/fvg/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/fvg_from_north_italy/" --idx_file="idx_test.pkl" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/cl_north_italy/checkpoint_3.pth" --checkpoint_reg="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/reg_north_italy/checkpoint_29.pth" --input_file="input_standard.pkl" --lat_dim=7 --lon_dim=7 --model_name_cl="Classifier_edges_test" --model_name_reg="Regressor_edges_test" --output_file="G_predictions_2016_test.pkl" --log_file="log.txt" --make_plots --img_extension="jpg"

# fvg
#python main.py --input_path="/leonardo_work/ICT23_ESP_0/SHARED/preprocessed/fvg/" --output_path="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/predictions/test_leonardo/" --idx_file="idx_test.pkl" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/test_cl_leo/checkpoint_20.pth" --checkpoint_reg="/leonardo_work/ICT23_ESP_0/vblasone/climate-DL/local_single/test_reg_leo/checkpoint_49.pth" --input_file="input_standard.pkl" --lat_dim=7 --lon_dim=7 --model_name_cl="Classifier_edges_test" --model_name_reg="Regressor_edges_test" --output_file="G_predictions_2016_test.pkl" --log_file="log.txt" --make_plots --img_extension="jpg"

#python main.py --input_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/data_north_preprocessed/" --output_path="/m100_work/ICT23_ESP_C/vblasone/climate-DL/predictions/test_north-from-fvg_2016_cl_e24_reg-e49_z_only/" --idx_file="idx_test.pkl" --graph_file_test="G_test.pkl" --subgraphs="subgraphs.pkl" --checkpoint_cl="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/cl_test_z_only/checkpoint_24.pth" --checkpoint_reg="/m100_work/ICT23_ESP_C/vblasone/climate-DL/local_single/reg_test_z_only/checkpoint_49.pth" --input_file="input_standard.pkl" --lat_dim=15 --lon_dim=31 --model_name_cl="Classifier_z_only_test" --model_name_reg="Regressor_z_only_test" --output_file="G_predictions_2016_test.pkl" --log_file="log.txt" --make_plots --img_extension="jpg"


