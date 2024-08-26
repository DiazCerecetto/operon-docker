export LC_ALL=C.UTF-8
export LANG=C.UTF-8
apt-get update --fix-missing && apt-get install -y \
    default-jdk \
    bzip2 \
    ca-certificates \
    curl \
    gcc \
    git \
    wget \
    vim \
    build-essential \
    jq 
pip install --upgrade pip
ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
conda env create -f /home/app/environment.yml
