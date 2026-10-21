#!/bin/sh
# Decompress every runs/**/*.xz next to itself, keeping the .xz (needs xz).
cd "$(dirname "$0")" && find . -name '*.xz' -exec xz -dk {} +
