# To be invoked via `sh train.sh`.
# Creates the necessary directory and copies settings to train the network with given hyperparameters.

# Creates results directory
DIRS=($(ls -v -d ./results/*))
idx=${DIRS[-1]}
idx="${idx##*/}"

new_dir=./results/$((${idx}+1))
mkdir ${new_dir}

DIRS=($(ls -d -v ./results/*))
dir=${DIRS[-1]}
echo "Configuring experiment in ${dir}"

# Copies the settings into the results directory
cp ./conf/global_settings.py $dir

# Logs the settings into a csv
python log_settings.py

# Queue the program into slurm
sbatch --export=ALL,dir_path=$dir --job-name=$((${idx}+1))_train program.sh
