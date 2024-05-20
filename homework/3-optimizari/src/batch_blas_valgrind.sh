#!/bin/bash
valgrind --tool=memcheck --leak-check=full ./tema3_blas ../input/input_valgrind > ../memory/blas.memory.txt