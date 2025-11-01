#!/usr/bin/env bash
set -euo pipefail

find source -type f \( -name "*.cu" -o -name "*.cuh" \) | while read -r file; do
    echo "Hipifying: $file"
    hipify-perl --inplace "$file"

    case "$file" in
        *.cu)
            new="${file%.cu}.hip"
            ;;
        *.cuh)
            new="${file%.cuh}.hip.h"
            ;;
    esac

    echo "Renaming: $file → $new"
    mv "$file" "$new"
done