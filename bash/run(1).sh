# !/bin/bash
for K in 8 10 15 11 12 13 14 15 80 75 70 69 68 67 66 65 64 63 62 61 60 59 58 57 56 55 54 53 52 51 50 49 45 35 30 20 15;
do
	for beta in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
	do
		for mu in 0.1 0.2 0.4 0.6 0.8 1.0 2 3 5 7 9;
		do
			for sigma in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
			do
				for lr in 4;
				do
					echo $1
					python execute.py --dataset $1 --mu ${mu} --lr 1e-${lr} --beta ${beta} --knn ${K} --sigma ${sigma} --sc 0
				done
			done
		done
	done
done