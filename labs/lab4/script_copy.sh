#!/bin/bash

rm -rf task* utils

for n in {1..5}; 
do
    mkdir task$n
    scp iustina.caramida@fep.grid.pub.ro:/export/home/acs/stud/i/iustina.caramida/l4/task$n/* task$n
done

mkdir utils
scp iustina.caramida@fep.grid.pub.ro:/export/home/acs/stud/i/iustina.caramida/l4/utils/* utils
