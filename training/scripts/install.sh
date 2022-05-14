sudo mkdir -p /mnt/disks/metapredictor-shards
sudo mount -o discard,defaults /dev/sdb /mnt/disks/metapredictor-shards
sudo mkdir -p /mnt/disks/imagenet
sudo mount -o discard,defaults,noload  /dev/sdc /mnt/disks/imagenet
sudo apt update --yes
sudo apt install --yes openexr libopenexr-dev

pip3 install jupyter_http_over_ws opencv-python matplotlib wandb scipy scikit-image
pip install -U --no-deps tensorflow_graphics tensorflow_addons typeguard efficientnet
sleep 5
pip3 install -q --upgrade wandb
sleep 5
pip3 install wandb
sleep 5
python3 -m wandb login 0b4e8997ed39f4534ca7cab9123adf1bcc897ec0

export PATH=~/.local/bin:$PATH