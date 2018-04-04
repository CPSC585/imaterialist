#!/bin/bash
set -e

function show_usage() {
    echo ""
    echo "Usage: $(basename $0) COMMANDS"
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

    CONTAINER=$(docker ps | grep "cpsc585/cluster" | awk '{print $1}')
    echo $(nvidia-docker exec $CONTAINER $@)
}

if (( $# < 1 )); then
    show_usage
    exit -1
fi

main "$@"
exit 0