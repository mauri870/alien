#!/usr/bin/env bash
set -euo pipefail

find source -type f \( -name "*.cu" -o -name "*.cuh" \) | while read -r file; do
    echo "Processing: $file"
    hipify-perl --inplace "$file"
done