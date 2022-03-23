MODEL_OUT="./finetuned_models/squad-2.0/wwm_cased_finetuned_squad/checkpoint_45k_bak/"

# python libs/transformers/examples/pytorch/question-answering/run_qa_beam_search.py \
#     --model_name_or_path $MODEL_OUT  \
#     --dataset_name squad_v2 \
#     --do_eval \
#     --version_2_with_negative \
#     --max_seq_length 384 \
#     --doc_stride 128 \
#     --output_dir $MODEL_OUT  \
#     --per_device_eval_batch_size=32  \

MODEL_OUT="./finetuned_models/squad-2.0/xlnetbasecased/"
python libs/transformers/examples/pytorch/question-answering/run_qa.py \
    --model_name_or_path $MODEL_OUT  \
    --dataset_name squad_v2 \
    --do_eval \
    --version_2_with_negative \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir $MODEL_OUT  \
    --per_device_eval_batch_size=8  \