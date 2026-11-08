#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: ./run_tests.sh <graph_path>"
    exit 1
fi

GRAPH=$1
PERCENTAGES=(2 5 10 12.5 25 20)

for P in "${PERCENTAGES[@]}"; do
    echo ""
    echo "========================================"
    echo "Running test for ${P}%"
    echo "========================================"

    ./../build/test_ssspd "$GRAPH" "$P"

    # if [ $? -ne 0 ]; then
    #     echo "Test failed for ${P}%"
    #     exit 1
    # fi
done

echo ""
echo "All tests completed."