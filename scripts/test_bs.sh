#!/bin/bash

#usage: INS=4 BS=32 bash test_bs.sh

TOTAL_CORES=$(lscpu | awk '/^Core\(s\) per socket:/ { print $4 }')
INSTANCE_NUM=${INS}
BATCH_SIZE=${BS}

cores_per_script=$((TOTAL_CORES / INSTANCE_NUM))

for i in $(seq 0 $((INSTANCE_NUM-1)))
do
   start_core=$((i * cores_per_script))
   end_core=$(((i+1) * cores_per_script - 1))

   numactl -l -C ${start_core}-${end_core} python test_bs.py ${BATCH_SIZE} &
done

wait
echo "All instances completed"
