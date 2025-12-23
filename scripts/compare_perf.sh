#!/bin/bash
# Compare performance between jax-rs and jax-js

set -e

echo "==================================================================="
echo "Performance Comparison: jax-rs vs jax-js"
echo "==================================================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Step 1: Building jax-rs in release mode..."
cargo build --release

echo ""
echo "Step 2: Running jax-rs benchmarks..."
cargo bench --bench parity_bench -- --output-format bencher | tee results-rs.txt

echo ""
echo "${YELLOW}Note: jax-js benchmarks require manual setup${NC}"
echo "To run jax-js benchmarks:"
echo "  1. cd _wb/jax-js-main"
echo "  2. npm install"
echo "  3. npm run bench"
echo "  4. Compare results manually"
echo ""

echo "jax-rs benchmark results saved to: results-rs.txt"
echo ""
echo "${GREEN}✓ jax-rs benchmarks complete${NC}"
