#!/bin/bash
apptainer exec $CONTAINER_IMAGE  \
make $TASK -f Makefile
