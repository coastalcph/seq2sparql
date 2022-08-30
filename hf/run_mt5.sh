#!/bin/bash
#normal cpu stuff: allocate cpus, memory
#SBATCH --ntasks=1 --cpus-per-task=6
#SBATCH -p gpu --gres=gpu:titanrtx:6 --mem=16GB
#SBATCH --time=100:00:00
#SBATCH --job-name=mt5-seq2sparql
#SBATCH --output=experiments-mt5-s2s.out

# source activate seq2sparql
echo "HOSTNAME: " `hostname`
echo "ALLOCATED GPU: " $CUDA_VISIBLE_DEVICES

# Train T5
# DO NOT USE FP16
declare -A langs
langs["en"]="English"
langs["he"]="Hebrew"
langs["kn"]="Kannada"
langs["zh"]="Chinese"
for split in mcd1 mcd2 mcd3 random_split; do   # mcd1 mcd2 mcd3
    for lang in en he kn zh; do
        python train_t5.py \
            --model_name_or_path google/mt5-base \
            --do_train \
            --do_eval \
            --do_predict \
            --evaluation_strategy steps \
            --eval_steps 500 \
            --save_strategy steps \
            --save_steps 500 \
            --logging_strategy steps \
            --logging_steps 100 \
            --save_total_limit 1 \
            --load_best_model_at_end True \
            --learning_rate 5e-4 \
            --num_train_epochs 10 \
            --seed 16 \
            --source_lang src \
            --target_lang tgt \
            --source_prefix "translate ${langs[$lang]} to Sparql: " \
            --train_file ../mcwq/translations/$split/train.$lang.json \
            --validation_file ../mcwq/translations/$split/dev.$lang.json \
            --test_file ../mcwq/translations/$split/test.$lang.json \
            --use_fast_tokenizer False \
            --max_source_length 256 \
            --max_target_length 512 \
            --output_dir ./models/mt5-base-$lang-$split \
            --per_device_train_batch_size=2 \
            --per_device_eval_batch_size=2 \
            --overwrite_output_dir \
            --predict_with_generate
    done
done

# predict
# turn off multi-gpu training in accelerate
EVALUATE_ON=test
declare -A langs
langs["en"]="English"
langs["he"]="Hebrew"
langs["kn"]="Kannada"
langs["zh"]="Chinese"
for split in mcd1 mcd2 mcd3 random_split; do
    for lang in en he kn zh; do  # he kn zh
        echo "Predicting on $split/$lang"
        python train_t5.py \
            --model_name_or_path ./models/mt5-base-$lang-$split \
            --do_predict \
            --evaluation_strategy steps \
            --eval_steps 500 \
            --save_strategy steps \
            --save_steps 500 \
            --logging_strategy steps \
            --logging_steps 500 \
            --save_total_limit 1 \
            --load_best_model_at_end \
            --learning_rate 5e-4 \
            --num_train_epochs 20 \
            --seed 16 \
            --source_lang src \
            --target_lang tgt \
            --source_prefix "translate ${langs[$lang]} to Sparql: " \
            --validation_file ../mcwq/translations/$split/dev.en.json \
            --test_file ../mcwq/translations/$split/test.$lang.json \
            --use_fast_tokenizer False \
            --max_source_length 256 \
            --max_target_length 512 \
            --output_dir ./models/mt5-base-$lang-$split \
            --per_device_train_batch_size=32 \
            --per_device_eval_batch_size=32 \
            --predict_with_generate
    done
done

# evaluate
cd ..
EVALUATE_ON=test
for split in random_split mcd1 mcd2 mcd3; do   # mcd1 mcd2 mcd3 random_split
    for lang in en he kn zh; do  # en he kn zh
        echo "Evaluating on $split/$lang"
        python -m evaluate_main --questions_path=mcwq/translations/$split/${EVALUATE_ON}.$lang.txt \
        --golden_answers_path=mcwq/translations/$split/${EVALUATE_ON}.$lang.sparql \
        --inferred_answers_path=hf/models/mt5-small-$lang-$split/generated_predictions_$lang.txt \
        --output_path=hf/models/mt5-small-$lang-$split/result_$lang.txt
    done
done
