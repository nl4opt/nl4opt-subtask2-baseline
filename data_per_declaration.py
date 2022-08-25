from constants import *
from collections import namedtuple
from utils import generate_decoder_inputs_outputs
import json
import torch
from torch.utils.data import DataLoader
import tqdm
from transformers import AutoTokenizer
from data import LPMappingDataset, instance_fields, batch_fields, Instance, Batch

class DeclarationMappingDataset(LPMappingDataset):

    def numberize(self, tokenizer, vocabs):
        """Numberize word pieces, labels, etcs.
        :param tokenizer: Bert tokenizer.
        :param vocabs (dict): a dict of vocabularies.
        """

        data = []
        for doc_id, content in self.data.items():  # TODO verify: is this a list or dict?
            document = content['document']
            order_mapping = content['order_mapping']

            orig_input_ids = tokenizer([document], max_length=self.max_length, truncation=True)['input_ids'][0]

            decoder_input_chunks = self.create_decoder_input_chunks(content, tokenizer)
            const_triggers = [START_OF_CONST_DIR + " " + x['text'].strip(" ") + " " + END_OF_CONST_DIR for x in content['spans'] if x['label'] == 'CONST_DIR' and 'text' in x]
            obj_triggers = [START_OF_OBJ_DIR + " " + x['text'].strip(" ") + " " + END_OF_OBJ_DIR for x in content['spans'] if x['label'] == 'OBJ_DIR' and 'text' in x]

            triggers = [tokenizer.encode(x)[1:-1] for x in obj_triggers + const_triggers]

            # print("decoder_input_chunks", decoder_input_chunks)

            for decoder_input, trigger in zip(decoder_input_chunks, triggers):

                # Declaration prompt trigger
                decl_trigger = trigger
                # TODO: wrap the trigger in <s>TRIGGER</s>
                # decl_trigger = [tokenizer.bos_token_id] + decoder_input[0] + [tokenizer.eos_token_id]
                pad_num = self.max_length - len(decl_trigger) - len(orig_input_ids)
                attn_mask = [1] * (len(decl_trigger) + len(orig_input_ids)) + [0] * pad_num
                input_ids = decl_trigger + orig_input_ids + [tokenizer.pad_token_id] * pad_num

                assert len(input_ids) == self.max_length, len(input_ids)
                input_tokens = tokenizer.decode(input_ids)
                instance = Instance(
                    doc_id=doc_id,
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    decoder_input_chunks=[decoder_input],
                    input_tokens=input_tokens,
                    document=document,
                    order_mapping=order_mapping
                )
                data.append(instance)
        self.data = data


if __name__ == '__main__':
    train_file = "data/samples/train-sample.jsonl"
    dev_file = "data/samples/dev-sample.jsonl"
    test_file = "data/samples/test-sample.jsonl"

    train_set = DeclarationMappingDataset(train_file, max_length=512, gpu=True)
    dev_set = DeclarationMappingDataset(dev_file, max_length=512, gpu=True)
    test_set = DeclarationMappingDataset(test_file, max_length=512, gpu=True)
    assert len(train_set) == 5
    assert len(dev_set) == 1
    assert len(test_set) == 2

    def test_example(example):
        assert all(k in example for k in [
            "document", "vars", "var_mentions", "params",
            "var_mention_to_first_var", "first_var_to_mentions",
            "obj_declaration", "const_declarations"
        ])

        # Objective and constraints
        assert all(k in example["obj_declaration"] for k in ["type", "name", "direction", "terms"])
        assert all(k in decl for k in ["type", "limit", "direction"] for decl in example["const_declarations"])

        # Variable labels
        assert 0 <= len(example["vars"]) <= len(example["var_mentions"])
        assert all(k in example["vars"] for k in example['first_var_to_mentions'])
        assert all(k in example["var_mentions"] for k in example['var_mention_to_first_var'])

        # objective
        assert all(k in example["var_mentions"] for k in example["obj_declaration"]["terms"].keys())

    # test single example
    example = dev_set['1657795738']
    assert example["obj_declaration"]["direction"] == "maximum"
    assert example["vars"] == ['hamburgers', 'hot dogs']
    assert example["var_mentions"] == ['hamburgers', 'hot dogs', 'hamburgers', 'hot dogs', 'hamburger', 'hot dog']
    assert example["obj_declaration"]["type"] == "objective"
    assert example["obj_declaration"]["name"] == "profit"
    test_example(example)

    # test all examples in dataset
    for doc_id, e in tqdm.tqdm(train_set.data.items()):
        test_example(e)
    for doc_id, e in tqdm.tqdm(dev_set.data.items()):
        test_example(e)
    for doc_id, e in tqdm.tqdm(test_set.data.items()):
        test_example(e)
    print("Tests DeclarationMappingDataset creation passed.")

    # test Numberize

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base",
                                              cache_dir="./bert")
    tokenizer.add_tokens(SPECIAL_TOKENS)
    train_set.numberize(tokenizer, vocabs={})
    dev_set.numberize(tokenizer, vocabs={})
    assert len(dev_set.data) == 6


    def test_decoder_input_chunks(dataset, index, expected_chunks):
        assert isinstance(dataset[index], Instance)
        decoder_input_chunks = dataset[index].decoder_input_chunks
        assert len(decoder_input_chunks) == 1
        assert sum(dataset[index].attention_mask) < len(dataset[0].input_ids)
        # assert dataset[index].input_tokens.startswith('<s>')
        assert '</s>' in dataset[index].input_tokens
        assert [tokenizer.decode(x) for x in decoder_input_chunks[0]] == expected_chunks


    test_decoder_input_chunks(dev_set, 0, [
        '<OBJ_DIR> maximum </OBJ_DIR>', '<OBJ_NAME> profit </OBJ_NAME>',
        ' [is] ',
        '<VAR> hamburger </VAR> [TIMES] <PARAM> 33 </PARAM>',
        '<VAR> hot dog </VAR> [TIMES] <PARAM> 21 </PARAM>'
    ])

    test_decoder_input_chunks(dev_set, 1, [
        '<CONST_DIR> at least </CONST_DIR>', '<LIMIT> 10 </LIMIT>',
        '<CONST_TYPE> [LOWER_BOUND] </CONST_TYPE>',
        ' [for] ',
        '<VAR> hamburgers </VAR>'
    ])

    test_decoder_input_chunks(dev_set, 2, [
        '<CONST_DIR> not cook more than </CONST_DIR>', '<LIMIT> 40 </LIMIT>',
        '<CONST_TYPE> [UPPER_BOUND] </CONST_TYPE>',
        ' [for] ',
        '<VAR> hamburgers </VAR>'
    ])

    test_decoder_input_chunks(dev_set, 3, [
        '<CONST_DIR> at least </CONST_DIR>', '<LIMIT> 30 </LIMIT>',
        '<CONST_TYPE> [LOWER_BOUND] </CONST_TYPE>',
        ' [for] ',
        '<VAR> hot dogs </VAR>'
    ])

    test_decoder_input_chunks(dev_set, 4, [
        '<CONST_DIR> not cook more than </CONST_DIR>', '<LIMIT> 70 </LIMIT>',
        '<CONST_TYPE> [UPPER_BOUND] </CONST_TYPE>',
        ' [for] ',
        '<VAR> hot dogs </VAR>'
    ])

    test_decoder_input_chunks(dev_set, 5, [
        '<CONST_DIR> not cook more than </CONST_DIR>',
        '<LIMIT> 90 </LIMIT>',
        '<CONST_TYPE> [SUM_CONSTRAINT] </CONST_TYPE>'
    ])


    print("Tests DeclarationMappingDataset numberize passed.")


    # Test data loaders

    train_loader = DataLoader(train_set, batch_size=2, shuffle=False, collate_fn=train_set.collate_fn)
    for i, batch in enumerate(train_loader):
        if i == 0:
            assert batch is not None
            assert tuple(batch.attention_masks.shape) == (2, 512)
            assert tuple(batch.input_ids.shape) == (2, 512)
            assert len(batch.decoder_input_chunks) == 2
            assert len(batch.document) == 2
            assert len(batch.input_tokens) == 2
            print("Tests LPMappingDataset collate_fn passed.")


            # Test the generate_decoder_inputs_outputs
            from unittest import mock
            model = mock.MagicMock()
            model.bert.config.name_or_path = "facebook/bart"
            model.bert.config.decoder_start_token_id = 9999
            input_outputs = generate_decoder_inputs_outputs(batch, tokenizer, model, True, 512)
        else:
            break
