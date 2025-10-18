#!/bin/bash
#SBATCH -J ghost_test
#SBATCH -o %x.out
#SBATCH -N 1
#SBATCH -n 15
#SBATCH -t 48:00:00
#SBATCH --nodelist=c3

ml python
python3 run_newton.py
