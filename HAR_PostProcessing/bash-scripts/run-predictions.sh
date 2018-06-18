#!/bin/sh
trap 'worker=`expr $worker - 1`' USR1  # free up a worker
worker=0  # current worker
num_workers=10  # maximum number of workers
for file in *.txt; do
    if [ $worker -lt $num_workers ]; then
        {   customScript -c 33 -I -file $file -a -v 55 > `basename $file .txt`.outtxt
            kill -USR1 $$ 2>/dev/null  # signal parent that we're free
        } &
        echo $worker/$num_worker $! $file  # feedback to caller
        worker=`expr $worker + 1`
    else
        wait # for a worker to finish
    fi
done