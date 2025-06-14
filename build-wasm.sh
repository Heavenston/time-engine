#!/bin/bash

set -e

mkdir -p dist
rm -rf dist/*
cargo build -p visualisation --release --target wasm32-unknown-unknown
cp -r public/* dist
if [ -x "$(command -v wasm-opt)" ]; then
  wasm-opt -O3 --strip-debug target/wasm32-unknown-unknown/release/visualisation.wasm -o dist/visualisation.wasm
else
  cp target/wasm32-unknown-unknown/release/visualisation.wasm dist/visualisation.wasm
fi
