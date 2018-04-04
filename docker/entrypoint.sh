#!/bin/bash
set -e

cpu=$(lscpu | grep "^CPU(s):" | awk '{ print $NF }')
sed -i 's/<cpus>/'$cpu'/g' /etc/slurm-llnl/slurm.conf

hostname=$(hostname)
sed -i 's/<hostname>/'$hostname'/g' /etc/slurm-llnl/slurm.conf

sockets=$(lscpu | grep "Socket(s):" | awk '{ print $NF }')
sed -i 's/<sockets>/'$sockets'/g' /etc/slurm-llnl/slurm.conf

corespersocket=$(lscpu | grep "Core(s) per socket:" | awk '{ print $NF }')
sed -i 's/<corespersocket>/'$corespersocket'/g' /etc/slurm-llnl/slurm.conf

threadspercore=$(lscpu | grep "Thread(s) per core:" | awk '{ print $NF }')
sed -i 's/<threadspercore>/'$threadspercore'/g' /etc/slurm-llnl/slurm.conf

gpucount=$(ls /proc/driver/nvidia/gpus | wc -l)
sed -i 's/<gpucount>/'$gpucount'/g' /etc/slurm-llnl/slurm.conf

if [[ $cpu == "0" ]]; then
    affinity="0"
else
    affinity="0-$((cpu-1))"
fi
for i in $(seq 1 $gpucount); do echo "Name=gpu File=/dev/nvidia$((i-1)) CPUs=$affinity" >> /etc/slurm-llnl/gres.conf; done

chmod -R 777 /var/log
chmod -R 777 /var/run
chown munge:munge /etc/munge/munge.key

echo ""
echo "***** STARTING SLURM *****"
echo ""
service munge start
service slurmctld start
service slurmd start
echo ""
echo "***** SLURM *****"
echo ""
sinfo

tail -f /dev/null
