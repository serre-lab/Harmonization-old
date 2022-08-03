# $name $zone $tpu $lr $lambda
#
# ./run.sh metapredictor-1 us-central1-a v3-8 0.1 1.0
#
echo "\n\n"
echo "\t Trying to start tpu vms: ($1) zone: ($2) tpu: ($3) (learning_rate=$4, lambda=$5, model=$6)"
echo "\n\n"

#if [ "$3" = "v4-8" ]; then
#    gcloud alpha compute tpus tpu-vm create $1 --zone=$2 --accelerator-type=$3 --version=v2-alpha-tpuv4
#else
#    gcloud alpha compute tpus tpu-vm create $1 --zone=$2 --accelerator-type=$3 --version=tpu-vm-tf-2.7.1
#fi

sleep 2

# attach disk
 gcloud alpha compute tpus tpu-vm attach-disk $1 --disk=metapred-ssd-shards --zone=$2 --mode=read-only
 gcloud alpha compute tpus tpu-vm attach-disk $1 --disk=imagenet-ssd-shards --zone=$2 --mode=read-only
# sleep 3

echo "\n\n"
echo "\t Trying to start tpu vms: ($1) zone: ($2) tpu: ($3) (learning_rate=$4, lambda=$5, model=$6)"
echo "\n\n"

# sending files
gcloud alpha compute tpus tpu-vm scp --recurse '../training/'* $1:~/ --zone=$2

sleep 3

# install and running the py script
install="bash scripts/install.sh"
pyrun="screen -dmS training bash -c 'python3 run.py $4 $5 $6'"
cmd="$install && $pyrun"

echo "\n Starting : $cmd"

gcloud alpha compute tpus tpu-vm ssh $1 --zone=$2 --command="$cmd"