# {
# "exact": 80.4177545691906,
# "f1": 84.07154997729623,
# "total": 11873,
# "HasAns_exact": 76.73751686909581,
# "HasAns_f1": 84.05558584352873,
# "HasAns_total": 5928,
# "NoAns_exact": 84.0874684608915,
# "NoAns_f1": 84.0874684608915,
# "NoAns_total": 5945
# }


export SQUAD_DIR="data/squad"
MODEL_OUT="./finetuned_models/squad-2.0/xlnetbasecased"

mkdir -p $MODEL_OUT

# python libs/transformers/examples/pytorch/question-answering/run_qa_beam_search.py \
#     --model_name_or_path xlnet-large-cased \
#     --dataset_name squad_v2 \
#     --do_train \
#     --do_eval \
#     --learning_rate 3e-5 \
#     --num_train_epochs 4 \
#     --max_seq_length 384 \
#     --doc_stride 128 \
#     --output_dir ${MODEL_OUT} --overwrite_output_dir \
#     --logging_dir ${MODEL_OUT}/tensorboard --logging_strategy steps \
#     --per_device_eval_batch_size=2  \
#     --per_device_train_batch_size=16   \
#     --save_steps 5000 --save_total_limit 1  --eval_steps 5000 


python libs/transformers/examples/pytorch/question-answering/run_qa_no_trainer.py \
    --model_name_or_path xlnet-base-cased \
    --dataset_name squad_v2 \
    --do_train true \
    --do_eval true \
    --learning_rate 3e-5 \
    --num_train_epochs 4 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ${MODEL_OUT}  \
    --per_device_eval_batch_size=2  \
    --per_device_train_batch_size=16   \
    # --save_steps 5000 --save_total_limit 1  --eval_steps 5000 

# python libs/transformers/examples/legacy/question-answering/run_squad.py \
#     --model_type xlnet \
#     --model_name_or_path xlnet-large-cased \
#     --do_train \
#     --do_eval \
#     --version_2_with_negative \
#     --train_file $SQUAD_DIR/train-v2.0.json \
#     --predict_file $SQUAD_DIR/dev-v2.0.json \
#     --learning_rate 3e-5 \
#     --num_train_epochs 4 \
#     --max_seq_length 384 \
#     --doc_stride 128 \
#     --output_dir ${MODEL_OUT}/wwm_cased_finetuned_squad/ \
#     --per_gpu_eval_batch_size=2  \
#     --per_gpu_train_batch_size=8   \
#     --save_steps 5000