#!/bin/bash
set -e

function show_usage() {
    echo ""
    echo "Usage: $(basename $0) --gpu=<GPU> COMMANDS"
    echo ""
    echo "   COMMANDS: any linux command"
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

    NV_GPU=$gpu CUDA_VISIBLE_DEVICES=$gpu nvidia-docker run -u $(id -u "AD\vkrishnamani"):$(id -g "AD\vkrishnamani") cpsc585/train $@
}

if (( $# < 1 )); then
    show_usage
    exit -1
fi

main "$@"
exit 0