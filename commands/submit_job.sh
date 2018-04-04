#!/bin/bash
set -e

function show_usage() {
    echo ""
    echo "Usage: $(basename $0) --gpu=<gpu> JOBCMD"
    echo ""
    echo "   --gpu_count: Number of GPUS."
    echo "                default: 1"
    echo ""
    echo "   JOBCMD: submit training job."
    echo ""
    echo "   --help: Show this help message."
    echo ""
}

source parser.sh

function main () {

    eval $(parse_params "$@")
    if [[ "$help" == "help" || "$h" == "h" ]]; then
        show_usage
        exit -1
    fi

    CONTAINER=$(docker ps | grep "cpsc585/cluster" | awk '{print $1}')
    GPUS=${gpu-1}
    shift;
    nvidia-docker exec $CONTAINER srun --gres=gpu:$GPUS --output=$output $@ 
}

if (( $# < 1 )); then
    show_usage
    exit -1
fi

main "$@"
exit 0