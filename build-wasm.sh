#!/bin/bash

set -e

mkdir -p dist
rm -rf dist/*
cargo build -p visualisation --release --target wasm32-unknown-unknown
cp -r public/* dist

if [ -x "$(command -v wasm-opt)" ]; then
  WASMOPT=wasm-opt
else
  dir=$(mktemp -d)
  wget https://github.com/WebAssembly/binaryen/releases/download/version_123/binaryen-version_123-x86_64-linux.tar.gz -O "$dir/binaryen.tar.gz"
  ( cd $dir; tar -xvf ./binaryen.tar.gz; )
  WASMOPT="$dir/binaryen-version_123/bin/wasm-opt"
  # cp target/wasm32-unknown-unknown/release/visualisation.wasm dist/visualisation.wasm
fi

$WASMOPT -O3 --strip-debug target/wasm32-unknown-unknown/release/visualisation.wasm -o dist/visualisation.wasm
