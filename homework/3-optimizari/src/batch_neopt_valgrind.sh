#!/bin/bash
valgrind --tool=memcheck --leak-check=full ./tema3_neopt ../input/input_valgrind > ../memory/neopt.memory.txt