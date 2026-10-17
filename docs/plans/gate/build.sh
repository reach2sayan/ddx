#!/bin/sh
set -e
R=/home/sayan/CLionProjects/ddx
B=$R/build/debug_jit
/usr/bin/g++-15 -DDDX_HAS_JIT -DDDX_JIT_DEFAULT_CONTRACT=1 -DDDX_JIT_DEFAULT_OPT=1 $DEFS -I$R/tests/rt -I$R/tests -I$R/include -I$B/include -isystem /opt/boost/include -g -std=gnu++23 -march=native -ffp-contract=fast -fno-math-errno -fno-exceptions -funwind-tables -Wall -Wextra -Wfatal-errors -o probe probe.cpp -L$B/src/rt -lddx -Wl,-rpath,$B/src/rt
