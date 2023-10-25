#!/bin/sh

SCRIPT_DIR=$(cd "$(dirname $0)"; pwd)
PROGS=$(ls ${SCRIPT_DIR}/*.py)
for p in ${PROGS}; do
  echo "executing: python3 ${p}"
  python3 ${p}
done
