
import tqdm
from torch.utils.data import DataLoader
from utils import *
from rouge import Rouge
import test_utils

import parsers
import scoring
from typing import Optional, Dict, List, Tuple

def collate_score_declarations(pred_texts: List[str], 
                                gold_texts: List[str], 
                                doc_ids: List[str], 
                                order_mappings: List[Dict], 
                                print_errors=True) -> float:
    current_id = ''
    current_pred_problem = ''
    current_gold_problem = ''
    pred_problems = []
    gold_problems = []
    mappings = []

    pred_canonicals = []
    gold_canonicals = []

    # converts an output into canonical form and returns the canonical form along with the order mapping
    # please ensure that the order mapping is consistent between pred and gold or the columns may be incorrect in the canonical form
    def parse_convert(output: str, order_mapping: Dict) -> parsers.CanonicalFormulation:
        parser = parsers.ModelOutputXMLParser(print_errors=print_errors)
        parsed = parser.parse(output, order_mapping)
        return parsers.convert_to_canonical(parsed)
    
    # we append as we do our predictions on the declaration level, i.e., we keep appending declarations until we get to the next problem
    # loop assumes that same doc_ids are contiguous
    # models that predict the entire formulation at once should not use this loop
    for pred, gold, doc_id, order_mapping in zip(pred_texts, gold_texts,doc_ids,order_mappings):
        if current_id != doc_id:
            # append order mapping of new problem
            mappings.append(order_mapping)
            current_id = doc_id
            if current_pred_problem and current_gold_problem:
                # append texts of previous problem
                gold_problems.append(current_gold_problem)
                pred_problems.append(current_pred_problem)
                current_pred_problem = ''
                current_gold_problem = ''
        
        current_pred_problem += pred
        current_gold_problem += gold

    # append texts for last problem, don't need to do for order mapping as it will already have been appended
    gold_problems.append(current_gold_problem)
    pred_problems.append(current_pred_problem)

    for pred, gold, order_mapping in zip(pred_problems, gold_problems, mappings):
        # use gold's order mapping in prediction for consistency in producing canonical form
        gold_canonical = parse_convert(gold, order_mapping)
        pred_canonical = parse_convert(pred, order_mapping)
        pred_canonicals.append(pred_canonical)
        gold_canonicals.append(gold_canonical)

    return scoring.overall_score(
        [x.objective for x in pred_canonicals],
        [x.constraints for x in pred_canonicals],
        [x.objective for x in gold_canonicals],
        [x.constraints for x in gold_canonicals],
    )



def evaluate(tokenizer, 
                model, 
                dataset, 
                epoch, 
                batch_num,
                use_gpu, 
                config, 
                tqdm_descr="Dataset", 
                ckpt_basename = "",
                print_errors=True):

    progress = tqdm.tqdm(total=batch_num, ncols=75,
                         desc='{} {}'.format(tqdm_descr, ckpt_basename))
    gold_outputs, pred_outputs, input_tokens, doc_ids, documents, order_mappings = [], [], [], [], [], []
    pred_texts, gold_texts, gold_pred_pairs = [], [], []
    metric = Rouge()
    measures = []
    for batch in DataLoader(dataset, batch_size=config.eval_batch_size,
                            shuffle=False, collate_fn=dataset.collate_fn):
        progress.update(1)
        outputs = model.predict(batch, tokenizer, epoch=epoch)
        decoder_inputs_outputs = generate_decoder_inputs_outputs(batch, tokenizer, model, use_gpu,
                                                                 config.max_position_embeddings)
        pred_outputs.extend(outputs['decoded_ids'].tolist())
        gold_outputs.extend(decoder_inputs_outputs['decoder_labels'].tolist())
        input_tokens.extend(batch.input_tokens)
        doc_ids.extend(batch.doc_ids)
        documents.extend(batch.document)
        order_mappings.extend(batch.order_mapping)
        pred_txt = [tokenizer.decode(x.tolist()) for x in outputs['decoded_ids']][0][4:]
        gold_txt = [tokenizer.decode(x.tolist()) for x in decoder_inputs_outputs['decoder_labels']][0]
        pred_txt, gold_txt = pred_txt[3:-4], gold_txt[3:-4]
        pred_texts.append(pred_txt)
        gold_texts.append(gold_txt)
        diff_metrics = {}
        measures.extend(diff_metrics)

        gold_pred_pairs.append({
            "gold": gold_txt,
            "pred": pred_txt,
            "document": batch.document[0],
            "rouge": diff_metrics
        })
    progress.close()
    accuracy = collate_score_declarations(pred_texts, gold_texts,doc_ids,order_mappings, print_errors)
    avg_metric = {}

    result = {
        'accuracy': accuracy,
        'pred_outputs': pred_outputs,
        'gold_outputs': gold_outputs,
        'input_tokens': input_tokens,
        'doc_ids': doc_ids,
        'documents': documents,
        'pred_texts': pred_texts,
        'gold_texts': gold_texts,
        'gold_pred_pairs': gold_pred_pairs,
        'rouge': avg_metric
    }

    return result
