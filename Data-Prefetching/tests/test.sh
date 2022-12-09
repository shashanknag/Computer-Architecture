./config.sh champsim_config.json
make
bin/champsim --warmup_instructions 50000000 --simulation_instructions 50000000 444.namd-120B.champsimtrace.xz 
bin/champsim --warmup_instructions 50000000 --simulation_instructions 50000000 437.leslie3d-134B.champsimtrace.xz
bin/champsim --warmup_instructions 50000000 --simulation_instructions 50000000 435.gromacs-111B.champsimtrace.xz
bin/champsim --warmup_instructions 50000000 --simulation_instructions 50000000 433.milc-127B.champsimtrace.xz
bin/champsim --warmup_instructions 50000000 --simulation_instructions 50000000 429.mcf-184B.champsimtrace.xz 
bin/champsim --warmup_instructions 50000000 --simulation_instructions 50000000 416.gamess-875B.champsimtrace.xz
bin/champsim --warmup_instructions 50000000 --simulation_instructions 50000000 410.bwaves-1963B.champsimtrace.xz
bin/champsim --warmup_instructions 50000000 --simulation_instructions 50000000 403.gcc-16B.champsimtrace.xz
bin/champsim --warmup_instructions 50000000 --simulation_instructions 50000000 401.bzip2-226B.champsimtrace.xz
bin/champsim --warmup_instructions 50000000 --simulation_instructions 50000000 400.perlbench-41B.champsimtrace.xz

