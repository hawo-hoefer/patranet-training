#!/bin/bash

dirs=(
  "cuka"
  "cukab"
)

subsets=(
  "train"
  "val"
  "test"
)


while [[ $# -gt 0 ]]; do
  case $1 in
    --clean)
      CLEAN=1
      shift
      ;;
    *)
      echo "Invalid Argument '$1'."
      echo "usage: ./generate.sh [opt]"
      echo "options:" 
      echo "    --clean      remove generated data"
      exit 1
      ;;
  esac
done


set -xeo pipefail

for dir in ${dirs[@]}; do
  ln -srf ./cif/ ./data/$dir/cif
  pushd "./data/$dir/"
  for subset in ${subsets[@]}; do
    echo $subset
    if [[ $CLEAN -eq 0 ]]; then
      yaxs "$subset.yaml" -o $subset --chunk-size=160000 --overwrite
    else
      rm -rvf $subset
    fi
  done
  popd
done
