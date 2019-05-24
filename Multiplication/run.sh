#!/bin/bash
#echo "Nombre de thread =2"

if [ $1 = "-r" ]
then
	env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/kmul_host --nthreads 2 > result.csv
else 
	env CL_CONTEXT_EMULATOR_DEVICE_ALTERA=1 ./bin/kmul_host --nthreads 2
fi 