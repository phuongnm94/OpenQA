import logging
from datasets import Dataset
from torch.utils.data import DataLoader
from datasets.table import InMemoryTable
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizerFast, default_data_collator, DataCollatorWithPadding
import torch
from collections import OrderedDict
import re
import pandas as pd
import numpy as np
from accelerate import Accelerator

from utils_qa import postprocess_qa_predictions, postprocess_qa_predictions_with_beam_search

logger = logging.getLogger(__name__)


# Post-processing:
def post_processing_function(args, examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=args.version_2_with_negative,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        null_score_diff_threshold=args.null_score_diff_threshold,
        output_dir=args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if args.version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    # references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    # return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    return formatted_predictions

def post_processing_function(args, examples, features, predictions, stage="eval", start_n_top=None, end_n_top=None):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions, scores_diff_json = postprocess_qa_predictions_with_beam_search(
        examples=examples,
        features=features,
        predictions=predictions,
        version_2_with_negative=args.version_2_with_negative,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        start_n_top=start_n_top,
        end_n_top=end_n_top,
        output_dir=args.output_dir,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    if args.version_2_with_negative:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": scores_diff_json[k]}
            for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

    # references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples]
    # return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    return formatted_predictions

def create_and_fill_np_array(start_or_end_logits, dataset, max_len):
    """
    Create and fill numpy array of size len_of_validation_data * max_length_of_output_tensor

    Args:
        start_or_end_logits(:obj:`tensor`):
            This is the output predictions of the model. We can only enter either start or end logits.
        eval_dataset: Evaluation dataset
        max_len(:obj:`int`):
            The maximum length of the output tensor. ( See the model.eval() part for more details )
    """

    step = 0
    # create a numpy array and fill it with -100.
    logits_concat = np.full((len(dataset), max_len), -100, dtype=np.float64)
    # Now since we have create an array now we will populate it with the outputs gathered using accelerator.gather
    for i, output_logit in enumerate(start_or_end_logits):  # populate columns
        # We have to fill it such that we have to take the whole tensor and replace it on the newly created array
        # And after every iteration we have to change the step

        batch_size = output_logit.shape[0]
        cols = output_logit.shape[1]

        if step + batch_size < len(dataset):
            logits_concat[step : step + batch_size, :cols] = output_logit
        else:
            logits_concat[step:, :cols] = output_logit[: len(dataset) - step]

        step += batch_size

    return logits_concat

class WikiQADataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, title, question, context_s):
        """
        """
        data = {'id': [str(i) for i in range(len(context_s))], 'title': [
            title]*len(context_s), 'question': [question]*len(context_s), 'context': context_s}
        df = pd.DataFrame(data=data)
        super().__init__(InMemoryTable.from_pandas(df=df))


class DocumentReader:
    def __init__(self, pretrained_model_name_or_path='bert-large-uncased', xlnet_model=False):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.READER_PATH = pretrained_model_name_or_path
        self.xlnet_model = xlnet_model
        if xlnet_model:
            self.train_args = torch.load(
                pretrained_model_name_or_path+"/train_args.torch")
            self.config = XLNetConfig.from_pretrained(self.READER_PATH)
            self.tokenizer = XLNetTokenizerFast.from_pretrained(
                self.READER_PATH)
            self.model = XLNetForQuestionAnswering.from_pretrained(self.READER_PATH,
                                                                   config=self.config,
                                                                   ).to(self.device)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.READER_PATH).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.READER_PATH)
        self.max_len = self.model.config.max_position_embeddings
        self.chunked = True

        # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
        self.accelerator = Accelerator()
        self.accelerator.wait_for_everyone()

        logger.info(self.accelerator.state)

    @staticmethod
    def prepare_validation_features(examples, question_column_name='question', context_column_name='context',
                                    tokenizer=None, doc_stride=None, max_seq_length=None, pad_to_max_length=None, pad_on_right=None):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip()
                                          for q in examples[question_column_name]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # The special tokens will help us build the p_mask (which indicates the tokens that can't be in answers).
        special_tokens = tokenized_examples.pop("special_tokens_mask")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        # We still provide the index of the CLS token and the p_mask to the model, but not the is_impossible label.
        tokenized_examples["cls_index"] = []
        tokenized_examples["p_mask"] = []

        for i, input_ids in enumerate(tokenized_examples["input_ids"]):
            # Find the CLS token in the input ids.
            cls_index = input_ids.index(tokenizer.cls_token_id)
            tokenized_examples["cls_index"].append(cls_index)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples["token_type_ids"][i]
            for k, s in enumerate(special_tokens[i]):
                if s:
                    sequence_ids[k] = 3
            context_idx = 1 if pad_on_right else 0

            # Build the p_mask: non special tokens and context gets 0.0, the others 1.0.
            tokenized_examples["p_mask"].append(
                [
                    0.0 if (not special_tokens[i][k] and s == context_idx) or k == cls_index else 1.0
                    for k, s in enumerate(sequence_ids)
                ]
            )

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_idx else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def _preprocess_text(self, question, text, intro_context, chunk_doc_min_size, max_toks):
        # text preprocessing
        if max_toks is None:
            max_toks = max(self.max_len-len(question.split(" ")) - 50, 0)
        self.text_s = [intro_context] + \
            self.text_chunk(text, chunk_doc_min_size, max_toks)
        self.text_s = [re.sub(r' {2,}', ' ', e.replace(
            "\n", " ").strip()) for e in self.text_s]
        self.text_s = [e for e in self.text_s if len(e) > 0] 

    def tokenize(self, question, text, intro_context='', chunk_doc_min_size=150, max_toks=None):
        self._preprocess_text(question, text, intro_context=intro_context, chunk_doc_min_size=chunk_doc_min_size, max_toks=max_toks)
        self.inputs = self.tokenizer.batch_encode_plus(
            [(question, text) for text in self.text_s], add_special_tokens=True, return_tensors="pt", padding=True, truncation=True)
        self.inputs = self.inputs.to(self.device)

    def tokenize_optimiz(self, question, text, intro_context='', chunk_doc_min_size=150, max_toks=None):
        # text preprocessing        
        self._preprocess_text(question, text, intro_context=intro_context, chunk_doc_min_size=chunk_doc_min_size, max_toks=max_toks)

        # using Dataset object
        self.dataset = WikiQADataset(
            title='', question=question, context_s=self.text_s)

        # multi processing by accelerate
        with self.accelerator.main_process_first():
            self.dataset_features = self.dataset.map(
                self.prepare_validation_features,
                batched=True,
                remove_columns=self.dataset.column_names,
                desc="Running tokenizer on query and Wiki page",
                fn_kwargs={'tokenizer': self.tokenizer,
                           'doc_stride': self.train_args.doc_stride,
                           'max_seq_length':  self.train_args.max_seq_length,
                           'pad_to_max_length': True,  #self.train_args.pad_to_max_length,
                           'pad_on_right': self.tokenizer.padding_side == "right"}
            )


    @staticmethod
    def text_chunk(doc, doc_min_size=60, max_token=400):
        docs = [""]
        for cur_doc in re.split(r'\n\n+', doc):
            docs[-1] = docs[-1] + " " + cur_doc
            last_doct_tokens = docs[-1].split(" ")
            len_last_doc = len(last_doct_tokens)
            if len_last_doc > doc_min_size or len_last_doc > max_token:
                if len_last_doc > max_token:
                    tmp_docs = []
                    for idx_chunk in range(0, len_last_doc, max_token):
                        tmp_docs.append(
                            " ".join(last_doct_tokens[idx_chunk:idx_chunk+max_token]))
                    docs = docs[:-1] + tmp_docs
                docs.append("")
        if len(docs[-1]) == 0:
            docs = docs[:-1]
        return docs


    def get_answer_optimiz(self):
    
        # DataLoaders creation:
        if self.train_args.pad_to_max_length:
            data_collator = default_data_collator
        else:
            data_collator = DataCollatorWithPadding(
                self.tokenizer, pad_to_multiple_of=(8 if self.accelerator.use_fp16 else None))
        predict_dataloader = DataLoader(
            self.dataset_features, collate_fn=data_collator, batch_size=self.train_args.per_device_eval_batch_size
        )

        logger.info("***** Running Prediction *****")
        logger.info(f"  Num examples = {len(self.dataset_features)}")
        logger.info(f"  Batch size = {self.train_args.per_device_eval_batch_size}")


        all_start_top_log_probs = []
        all_start_top_index = []
        all_end_top_log_probs = []
        all_end_top_index = []
        all_cls_logits = []
        for step, batch in enumerate(predict_dataloader):
            batch = batch.to(self.device)
            with torch.no_grad():
                outputs = self.model(**batch)
                start_top_log_probs = outputs.start_top_log_probs
                start_top_index = outputs.start_top_index
                end_top_log_probs = outputs.end_top_log_probs
                end_top_index = outputs.end_top_index
                cls_logits = outputs.cls_logits

                if not self.train_args.pad_to_max_length:  # necessary to pad predictions and labels for being gathered
                    start_top_log_probs = self.accelerator.pad_across_processes(start_top_log_probs, dim=1, pad_index=-100)
                    start_top_index = self.accelerator.pad_across_processes(start_top_index, dim=1, pad_index=-100)
                    end_top_log_probs = self.accelerator.pad_across_processes(end_top_log_probs, dim=1, pad_index=-100)
                    end_top_index = self.accelerator.pad_across_processes(end_top_index, dim=1, pad_index=-100)
                    cls_logits = self.accelerator.pad_across_processes(cls_logits, dim=1, pad_index=-100)

                all_start_top_log_probs.append(self.accelerator.gather(start_top_log_probs).cpu().numpy())
                all_start_top_index.append(self.accelerator.gather(start_top_index).cpu().numpy())
                all_end_top_log_probs.append(self.accelerator.gather(end_top_log_probs).cpu().numpy())
                all_end_top_index.append(self.accelerator.gather(end_top_index).cpu().numpy())
                all_cls_logits.append(self.accelerator.gather(cls_logits).cpu().numpy())

        max_len = max([x.shape[1] for x in all_end_top_log_probs])  # Get the max_length of the tensor

        # concatenate all numpy arrays collected above
        start_top_log_probs_concat = create_and_fill_np_array(all_start_top_log_probs, self.dataset_features, max_len)
        start_top_index_concat = create_and_fill_np_array(all_start_top_index, self.dataset_features, max_len)
        end_top_log_probs_concat = create_and_fill_np_array(all_end_top_log_probs, self.dataset_features, max_len)
        end_top_index_concat = create_and_fill_np_array(all_end_top_index, self.dataset_features, max_len)
        cls_logits_concat = np.concatenate(all_cls_logits, axis=0)

        # delete the list of numpy arrays
        del start_top_log_probs
        del start_top_index
        del end_top_log_probs
        del end_top_index
        del cls_logits

        outputs_numpy = (
            start_top_log_probs_concat,
            start_top_index_concat,
            end_top_log_probs_concat,
            end_top_index_concat,
            cls_logits_concat,
        )

        prediction = post_processing_function(self.train_args, self.dataset, self.dataset_features, outputs_numpy,
        self.model.config.start_n_top, self.model.config.end_n_top)
        predict_metric = metric.compute(predictions=prediction.predictions, references=prediction.label_ids)
        logger.info(f"Predict metrics: {predict_metric}")

    def get_answer(self):
        answer = ''
        detail_result = []
        output_reader = self.model(**self.inputs)

        for i_sample in range(len(self.text_s)):
            if self.xlnet_model:
                answer_start, answer_end = output_reader.start_top_index[i_sample][
                    0], output_reader.end_top_index[i_sample][0]
                prob = output_reader.start_top_log_probs[i_sample][0] * \
                    output_reader.end_top_log_probs[i_sample][0]
            else:
                answer_start_scores, answer_end_scores = output_reader.start_logits[
                    i_sample], output_reader.end_logits[i_sample]

                answer_start = torch.argmax(answer_start_scores)
                prob = torch.softmax(answer_start_scores, dim=0)[answer_start]
                answer_end = torch.argmax(answer_end_scores) + 1
                prob = prob*torch.softmax(answer_end_scores, dim=0)[answer_end]

            ans = self.convert_ids_to_string(
                self.inputs['input_ids'][i_sample][answer_start:answer_end])
            if ans != '[CLS]' and ans != "<s>" and answer_start <= answer_end:
                answer += ans + " / "
                context_left = self.convert_ids_to_string(
                    self.inputs['input_ids'][i_sample][:answer_start]).replace(
                    "<pad>", "").replace("[PAD]", "")
                context_right = self.convert_ids_to_string(
                    self.inputs['input_ids'][i_sample][answer_end:]).replace(
                    "<pad>", "").replace("[PAD]", "")
                detail_result.append({'answer': ans.strip(), 'a_start': answer_start.item(), 'a_end': answer_end.item(),
                                      'prob': float(prob.item()),
                                      'context': "{} ____[{}]____ {}".format(
                                          context_left, ans, context_right)})
        detail_result.sort(key=lambda x: x['prob'], reverse=True)
        if len(detail_result) > 0:
            answer = detail_result[0]['answer']
        return answer.strip(), detail_result

    def convert_ids_to_string(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids))


if __name__ == "__main__":
    question = "Who ruled Macedonia"

    context = """Macedonia was an ancient kingdom on the periphery of Archaic and Classical Greece, 
    and later the dominant state of Hellenistic Greece. The kingdom was founded and initially ruled 
    by the Argead dynasty, followed by the Antipatrid and Antigonid dynasties. Home to the ancient 
    Macedonians, it originated on the northeastern part of the Greek peninsula. Before the 4th 
    century BC, it was a small kingdom outside of the area dominated by the city-states of Athens, 
    Sparta and Thebes, and briefly subordinate to Achaemenid Persia. We tried our model on a question paired with a short passage, but what if we want to retrieve an answer from a longer document? A typical Wikipedia page is much longer than the example above, and we need to do a bit of massaging before we can use our model on longer contexts."""

    question = "what is your price?"

    context = """hello! I am glad to see you. I am a Pepper robot, an AI robot in JAIST. My price is around 1800 $, but my benefit is unccountable.
        I was born on 1 Jan 2022 at JAIST. I was born to support human search information on the Wikipedia Knowledge Base. Researchers at JAIST created me with their love. I can support human search information on the Wikipedia Knowledge Base.  I love you. 
    """
    # reader = DocumentReader("deepset/bert-base-cased-squad2")
    reader = DocumentReader(
        "finetuned_models/squad-2.0/wwm_cased_finetuned_squad", xlnet_model=True)

    reader.tokenize_optimiz(question, context, max_toks=200)
    results_reader = reader.get_answer_optimiz()
    import json
    print(json.dumps(results_reader[1], indent=2))
