pip3 install --yes jupyter jupyter_http_over_ws opencv-python matplotlib
sudo apt install  --yes openexr libopenexr-dev
export PATH=~/.local/bin:$PATH
python3 -m jupyter serverextension enable --py jupyter_http_over_ws
python3 -m jupyter notebook --NotebookApp.allow_origin='https://colab.research.google.com'  --port=7777 --NotebookApp.port_retries=0  --no-browser
# gcloud alpha compute tpus tpu-vm ssh metapredictor-1-tf --zone=us-central1-a -- -L 7777:localhost:7777