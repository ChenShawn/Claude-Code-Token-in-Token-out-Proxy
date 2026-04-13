#!/bin/bash
set -x

cd /claude-proxy
bash run_serving.sh

# In theory run_serving.sh will run permanently
echo "ERROR exit: $?"
