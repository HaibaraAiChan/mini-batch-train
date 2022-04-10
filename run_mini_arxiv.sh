#!/bin/bash

File=mini_batch_train.py

epoch=5
Aggre=lstm
model=sage
seed=1236 
setseed=True
lr=0.01
dropout=0.5

layers=3
Data=ogbn-arxiv

hidden=64
run=1
fan_out_list=( 200,3,5 )
# fan_out_list=( 5,10,15 )
# fan_out_list=( 25,35,40 )
batch_size=(45471 22736 11368)
Aggre=mean
nb=1
for fan_out in ${fan_out_list[@]}
do
        for bs in ${batch_size[@]}
        do
                nb=$(($nb*2))
                python $File \
                --dataset $Data \
                --aggre $Aggre \
                --seed $seed \
                --setseed $setseed \
                --batch-size $bs \
                --lr $lr \
                --num-runs $run \
                --num-epochs $epoch \
                --num-layers $layers \
                --num-hidden $hidden \
                --dropout $dropout \
                --fan-out $fan_out \
                > ./logs/sage/${Agree}/epoch_${epoch}/${Data}_${Aggre}_${seed}_la_${layers}_fo_${fan_out}_nb_${nb}_run_${run}_ep_${epoch}.log
        done
done

