#!/bin/bash
set -e

function show_usage() {
    echo ""
    echo "Usage: $(basename $0) JOBCMD"
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

    CONTAINER=$(docker images | grep "cpsc585/base" | awk '{print $3}')
    docker run -u $(id -u "AD\vkrishnamani"):$(id -g "AD\vkrishnamani") -v $(pwd):/workspace $CONTAINER $@
}

if (( $# < 1 )); then
    show_usage
    exit -1
fi

main "$@"
exit 0