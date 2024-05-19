cd 3-optimizari/src/
ls
cat solver_blas.c 
valgrind –tool=memcheck –leak-check=full tema3_blas ../input/input_valgrind 
sudo valgrind –tool=memcheck –leak-check=full tema3_blas ../input/input_valgrind 
valgrind-listener –tool=memcheck –leak-check=full tema3_blas ../input/input_valgrind 
