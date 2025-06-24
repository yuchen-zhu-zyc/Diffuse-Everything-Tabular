dataset="default"
model_idx="0029999"
exp_dir="./exp/${dataset}/001"

for i in $(seq 1 20)
do

	python sample.py --dataname ${dataset} --ckpt_path ${exp_dir}/model_${model_idx}.pt --save_path ${exp_dir}/

	python eval/eval_mle.py --dataname ${dataset} --syn_path ${exp_dir}/samples_af_inv_${model_idx}.csv >> ${exp_dir}/mle_out

done

# results for each run would be saved in mle_out file. Then simply take the average and std of the results.
