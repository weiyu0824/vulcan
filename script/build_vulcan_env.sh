conda create -y -n vulcan python=3.10
conda init
. /opt/miniconda/etc/profile.d/conda.sh
echo '. /opt/miniconda/etc/profile.d/conda.sh' >> "$HOME/.bashrc"
conda activate vulcan 
