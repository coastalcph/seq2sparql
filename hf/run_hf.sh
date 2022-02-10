#!/bin/bash

#SBATCH --job-name=seq2sparql
#SBATCH --ntasks=1
#SBATCH --output=./logs/log.out
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:titanx
#SBATCH --mem=2GB

# source activate seq2sparql
echo "HOSTNAME: " `hostname`
echo "ALLOCATED GPU: " $CUDA_VISIBLE_DEVICES

# create datasets in correct format
# python make_hf_dataset.py mcd1
# python make_hf_dataset.py mcd2
# python make_hf_dataset.py mcd3
# python make_hf_dataset.py random_split

# MODEL_NAME=mbert-bbase
# EVALUATE_ON=test

# train on english data
# for split in mcd1 mcd2 mcd3 random_split; do
#     # train tokenizers only on english data
#     # python train_tokenizers.py --data_split=$split --lang=en --files ../data/hf_data/cwq_en/$split/train.json ../data/hf_data/cwq_en/$split/dev.json ../data/hf_data/cwq_en/$split/test.json
#     # train and evaluate model only on english data
#     accelerate launch train_seq2seq.py --per_device_train_batch_size=128 --per_device_eval_batch_size=128 \
#         --train_file=../data/hf_data/cwq_en/$split/train.json --validation_file=../data/hf_data/cwq_en/$split/dev.json \
#         --test_file=../data/hf_data/cwq_en/$split/test.json --output_dir=./models/${MODEL_NAME}_$split --num_train_epochs=100 \
#         --data_split=$split --pretrained_encoder --pretrained_encoder_freeze_emb # --only_eval # --eval_all
# done

# train on non-english data
# for split in mcd1 mcd2 mcd3 random_split; do
#     for lang in he kn zh; do
#         echo "Training on $split/$lang"
#         python train_tokenizers.py --data_split=$split --lang=$lang --files ../data/translations/$split/train.$lang.json ../data/translations/$split/dev.$lang.json ../data/translations/$split/test.$lang.json
#         accelerate launch train_seq2seq.py --per_device_train_batch_size=128 --per_device_eval_batch_size=128 \
#             --train_file=../data/translations/$split/train.$lang.json --validation_file=../data/translations/$split/dev.$lang.json \
#             --test_file=../data/translations/$split/test.$lang.json --output_dir=./models/mbert-bbase_${split}_$lang --num_train_epochs=100 \
#             --data_split=$split --pretrained_encoder --train_lang=$lang --eval_lang=$lang --pretrained_encoder_freeze_emb # --only_eval --eval_all
#     done
# done

# predict
# turn off multi-gpu training in accelerate
# for split in mcd1 mcd2 mcd3 random_split; do
#     for lang in en he kn zh; do
#         echo "Predicting on $split/$lang"
#         accelerate launch train_seq2seq.py --per_device_train_batch_size=128 --per_device_eval_batch_size=128 \
#             --train_file=../data/translations/$split/train.en.json --validation_file=../data/translations/$split/dev.$lang.json \
#             --test_file=../data/translations/$split/${EVALUATE_ON}.$lang.json --output_dir=./models/${MODEL_NAME}_$split --num_train_epochs=100 \
#             --data_split=$split --eval_lang=$lang --pretrained_encoder --pretrained_encoder_freeze_emb --only_eval # --eval_all
#     done
# done

# evaluate
# cd ..
# for split in mcd1 mcd2 mcd3 random_split; do
#     for lang in en he kn zh; do
#         echo "Evaluating on $split/$lang"
#         python -m evaluate_main --questions_path=./data/translations/$split/${EVALUATE_ON}.$lang.txt \
#         --golden_answers_path=./data/translations/$split/${EVALUATE_ON}.$lang.sparql \
#         --inferred_answers_path=hf/models/${MODEL_NAME}_$split/predictions_$lang.txt \
#         --output_path=hf/models/${MODEL_NAME}_$split/result_$lang.txt
#     done
# done
# cd hf/

# Train T5
# DO NOT USE FP16
declare -A langs
langs["en"]="English"
langs["he"]="Hebrew"
langs["kn"]="Kannada"
langs["zh"]="Chinese"
for split in random_split; do   # mcd1 mcd2 mcd3 
    for lang in he kn zh; do
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
            --learning_rate 3e-5 \
            --num_train_epochs 10 \
            --seed 42 \
            --source_lang src \
            --target_lang tgt \
            --source_prefix "translate ${langs[$lang]} to Sparql: " \
            --train_file ../data/translations/$split/train.$lang.json \
            --validation_file ../data/translations/$split/dev.$lang.json \
            --test_file ../data/translations/$split/test.$lang.json \
            --use_fast_tokenizer False \
            --max_source_length 128 \
            --max_target_length 256 \
            --output_dir ./models/mt5-base-256-$split-$lang \
            --per_device_train_batch_size=4 \
            --per_device_eval_batch_size=4 \
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
    for lang in en; do  # he kn zh
        echo "Predicting on $split/$lang"
        python train_t5.py \
            --model_name_or_path ./models/t5-base-$split \
            --do_predict \
            --evaluation_strategy steps \
            --eval_steps 500 \
            --save_strategy steps \
            --save_steps 500 \
            --logging_strategy steps \
            --logging_steps 500 \
            --save_total_limit 1 \
            --load_best_model_at_end \
            --learning_rate 3e-5 \
            --num_train_epochs 20 \
            --seed 42 \
            --source_lang src \
            --target_lang tgt \
            --source_prefix "translate ${langs[$lang]} to Sparql: " \
            --validation_file ../data/translations/$split/dev.en.json \
            --test_file ../data/translations/$split/test.$lang.json \
            --use_fast_tokenizer False \
            --max_source_length 128 \
            --max_target_length 128 \
            --output_dir ./models/t5-base-$split \
            --per_device_train_batch_size=32 \
            --per_device_eval_batch_size=32 \
            --predict_with_generate
    done
done

# evaluate
EVALUATE_ON=test
for split in mcd1 mcd2 mcd3 random_split; do   # mcd1 mcd2 mcd3 random_split 
    for lang in en he kn zh; do  # en he kn zh
        echo "Evaluating on $split/$lang"
        python -m evaluate_main --questions_path=./data/translations/$split/${EVALUATE_ON}.$lang.txt \
        --golden_answers_path=./data/translations/$split/${EVALUATE_ON}.$lang.sparql \
        --inferred_answers_path=hf/models/mt5-base-256-$split-$lang/generated_predictions_$lang.txt \
        --output_path=hf/models/mbert-bbase_${split}_$lang/result_$lang.txt
    done
done

