# FPGA-CPU-DEEP-NEURAL
## First Part : Matrix multiplication using OpenCL
We using Quartus

## Second Part : Algorithm of Deep Neural#
g++  -O2 -D__USE_XOPEN2K8 -Wall -IALSDK/include -I/usr/local/include -DHAVE_CONFIG_H -DTESTB -g -L/opt/aalsdk/lib  -L/usr/local/lib  -fPIC -Ihost/inc -I../common/inc -I../extlibs/inc \
                -I/home/guillaume/altera/16.0/hld/host/include  ../common/src/AOCLUtils/opencl.cpp ../common/src/AOCLUtils/options.cpp host/src/main.cpp -L/home/guillaume/altera/16.0/hld/host/linux64/lib -Wl,--no-as-needed -lalteracl -lstdc++ -lelf  \
                 \
                 \
                -o bin/kmul_host 