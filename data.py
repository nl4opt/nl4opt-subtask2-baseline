from torch.utils.data import Dataset
from constants import SPECIAL_TOKENS
from collections import namedtuple
from utils import token2sub_tokens, format_typed_mention, generate_decoder_inputs_outputs
import json
import torch
from torch.utils.data import DataLoader
import tqdm
from transformers import AutoTokenizer

instance_fields = [
    'doc_id', 'input_ids', 'attention_mask', 'decoder_input_chunks', 'input_tokens', 'document', 'order_mapping'
]

batch_fields = [
    'doc_ids', 'input_ids', 'attention_masks', 'decoder_input_chunks', 'input_tokens', 'document', 'order_mapping'
]

Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))


class LPMappingDataset(Dataset):
    def __init__(self, path, max_length=128, gpu=False, no_prompt=False):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        :param ignore_title (bool): Ignore sentences that are titles (default=False).
        """
        self.path = path
        self.data = []
        self.max_length = max_length
        self.gpu = gpu
        self.no_prompt = no_prompt

        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        """Load data from file."""
        with open(self.path, 'r', encoding='utf-8') as f:
            json_lines = [json.loads(line) for line in f]
            data = {k:v for x in json_lines for k,v in x.items()}
            self.data = data

    def create_decoder_input_chunks(self, example, tokenizer):

        obj_declaration = [example['obj_declaration']] if example['obj_declaration'] else []
        templates = obj_declaration + example['const_declarations']

        # Bart uses the eos_token_id as the starting token for decoder_input_ids generation.
        # If past_key_values is used, optionally only the last decoder_input_ids have to be input (see past_key_values)
        res = []
        for template in templates:

            # convert template to list of typed chunks
            typed_fragments = format_typed_mention(template)
            encoded_typed_fragments = []
            for fragment in typed_fragments:
                assert isinstance(fragment, list)
                # entity = []
                encoded_fragment = []
                for entity_token in fragment:
                    encoded_fragment += token2sub_tokens(tokenizer, entity_token)
                encoded_typed_fragments.append(encoded_fragment)
            res.append(encoded_typed_fragments)
        return res

    def numberize(self, tokenizer, vocabs):
        """Numberize word pieces, labels, etcs.
        :param tokenizer: Bert tokenizer.
        :param vocabs (dict): a dict of vocabularies.
        """

        data = []
        for doc_id, content in self.data.items():  # TODO verify: is this a list or dict?
            document = content['document']
            order_mapping = content.get('order_mapping', None)

            input_ids = tokenizer([document], max_length=self.max_length, truncation=True)['input_ids'][0]

            pad_num = self.max_length - len(input_ids)
            attn_mask = [1] * len(input_ids) + [0] * pad_num
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_num

            decoder_input_chunks = self.create_decoder_input_chunks(content, tokenizer)

            assert len(input_ids) == self.max_length, len(input_ids)

            input_tokens = tokenizer.decode(input_ids)
            # print("decoder_input_chunks", decoder_input_chunks)
            instance = Instance(
                doc_id=doc_id,
                input_ids=input_ids,
                attention_mask=attn_mask,
                decoder_input_chunks=decoder_input_chunks,
                input_tokens=input_tokens,
                document=document,
                order_mapping=order_mapping
            )
            data.append(instance)
        self.data = data

    def collate_fn(self, batch):
        batch_input_ids = []
        batch_attention_masks = []
        batch_decoder_input_chunks = []
        batch_input_tokens = []
        batch_document = []
        batch_order_mapping = []

        doc_ids = [inst.doc_id for inst in batch]

        for inst in batch:
            batch_input_ids.append(inst.input_ids)
            batch_attention_masks.append(inst.attention_mask)
            batch_decoder_input_chunks.append(inst.decoder_input_chunks)
            batch_input_tokens.append(inst.input_tokens)
            batch_document.append(inst.document)
            batch_order_mapping.append(inst.order_mapping)

        if self.gpu:
            batch_input_ids = torch.cuda.LongTensor(batch_input_ids)
            batch_attention_masks = torch.cuda.FloatTensor(batch_attention_masks)

        else:
            batch_input_ids = torch.LongTensor(batch_input_ids)
            batch_attention_masks = torch.FloatTensor(batch_attention_masks)

        return Batch(
            doc_ids=doc_ids,
            input_ids=batch_input_ids,
            attention_masks=batch_attention_masks,
            decoder_input_chunks=batch_decoder_input_chunks,
            input_tokens=batch_input_tokens,
            document=batch_document,
            order_mapping=batch_order_mapping
        )


if __name__ == '__main__':
    train_file = "data/samples/train-sample.jsonl"
    dev_file = "data/samples/dev-sample.jsonl"
    test_file = "data/samples/test-sample.jsonl"

    train_set = LPMappingDataset(train_file, max_length=512, gpu=True)
    dev_set = LPMappingDataset(dev_file, max_length=512, gpu=True)
    test_set = LPMappingDataset(test_file, max_length=512, gpu=True)
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
    print("Tests LPMappingDataset creation passed.")

    # test Numberize

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base",
                                              cache_dir="./bert")
    tokenizer.add_tokens(SPECIAL_TOKENS)
    train_set.numberize(tokenizer, vocabs={})
    dev_set.numberize(tokenizer, vocabs={})
    assert len(dev_set.data) == 1
    assert isinstance(dev_set[0], Instance)

    decoder_input_chunks = dev_set[0].decoder_input_chunks
    assert len(decoder_input_chunks) == 6
    assert sum(dev_set[0].attention_mask) < len(dev_set[0].input_ids)
    assert dev_set[0].input_tokens.startswith('<s>')
    assert '</s>' in dev_set[0].input_tokens

    assert [tokenizer.decode(x) for x in decoder_input_chunks[0]] == [
        '<OBJ_DIR> maximum </OBJ_DIR>', '<OBJ_NAME profit </OBJ_NAME>',
        ' [is] ',
        '<VAR> hamburger </VAR> [TIMES] <PARAM> 33 </PARAM>',
        '<VAR> hot dog </VAR> [TIMES] <PARAM> 21 </PARAM>'
    ]

    assert [tokenizer.decode(x) for x in decoder_input_chunks[1]] == [
        '<CONST_DIR> at least </CONST_DIR>', '<LIMIT> 10 </LIMIT>',
        '<CONST_TYPE> [LOWER_BOUND] </CONST_TYPE>',
        ' [for] ',
        '<VAR> hamburgers </VAR>'
    ]

    assert [tokenizer.decode(x) for x in decoder_input_chunks[2]] == [
        '<CONST_DIR> not cook more than </CONST_DIR>', '<LIMIT> 40 </LIMIT>',
        '<CONST_TYPE> [UPPER_BOUND] </CONST_TYPE>',
        ' [for] ',
        '<VAR> hamburgers </VAR>'
    ]
    assert [tokenizer.decode(x) for x in decoder_input_chunks[3]] == [
        '<CONST_DIR> at least </CONST_DIR>', '<LIMIT> 30 </LIMIT>',
        '<CONST_TYPE> [LOWER_BOUND] </CONST_TYPE>',
        ' [for] ',
        '<VAR> hot dogs </VAR>'
    ]
    assert [tokenizer.decode(x) for x in decoder_input_chunks[4]] == [
        '<CONST_DIR> not cook more than </CONST_DIR>', '<LIMIT> 70 </LIMIT>',
        '<CONST_TYPE> [UPPER_BOUND] </CONST_TYPE>',
        ' [for] ',
        '<VAR> hot dogs </VAR>']

    assert [tokenizer.decode(x) for x in decoder_input_chunks[5]] == [
        '<CONST_DIR> not cook more than </CONST_DIR>',
        '<LIMIT> 90 </LIMIT>',
        '<CONST_TYPE> [SUM_CONSTRAINT] </CONST_TYPE>'
    ]

    print("Tests LPMappingDataset numerize passed.")


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
