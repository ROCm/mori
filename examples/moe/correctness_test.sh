
thisScriptPath=$(dirname $0)
execPath=$thisScriptPath/../../build/examples/test_dispatch_combine
echo $execPath
# ------------------------------------------------------------------------------------------------ #
#                                          Inra-Node Test                                          #
# ------------------------------------------------------------------------------------------------ #
# Test 2 Rank
# mpirun -np 2 --allow-run-as-root $execPath 4096 1 2 2 4 4
# mpirun -np 2 --allow-run-as-root $execPath 4096 1 4 4 4 4
# mpirun -np 2 --allow-run-as-root $execPath 4096 7 2 2 4 4
# mpirun -np 2 --allow-run-as-root $execPath 4096 32 4 4 4 4
# mpirun -np 2 --allow-run-as-root $execPath 4096 128 4 4 4 4
# mpirun -np 2 --allow-run-as-root $execPath 4096 128 4 4 4 8 # bug encountered at this case

# # Test 4 Rank
mpirun -np 4 --allow-run-as-root $execPath 4096 1 2 2 4 4
mpirun -np 4 --allow-run-as-root $execPath 4096 1 4 4 4 4
mpirun -np 4 --allow-run-as-root $execPath 4096 7 2 2 4 4
mpirun -np 4 --allow-run-as-root $execPath 4096 32 4 4 4 4
mpirun -np 4 --allow-run-as-root $execPath 4096 128 4 4 4 4
mpirun -np 4 --allow-run-as-root $execPath 4096 128 4 4 4 8

# # Test 8 Rank
mpirun -np 2 --allow-run-as-root $execPath 4096 1 2 2 4 4
mpirun -np 2 --allow-run-as-root $execPath 4096 1 4 4 4 4
mpirun -np 4 --allow-run-as-root $execPath 4096 7 2 2 4 4
mpirun -np 8 --allow-run-as-root $execPath 4096 32 4 4 4 4
mpirun -np 8 --allow-run-as-root $execPath 4096 128 4 4 4 4
mpirun -np 8 --allow-run-as-root $execPath 4096 128 4 4 4 8