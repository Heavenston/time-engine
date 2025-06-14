#!/bin/bash

set -e

mkdir -p dist
rm -rf dist/*
cargo build -p visualisation --release --target wasm32-unknown-unknown
cp -r public/* dist
wasm-opt -O3 --strip-debug target/wasm32-unknown-unknown/release/visualisation.wasm -o dist/visualisation.wasm
