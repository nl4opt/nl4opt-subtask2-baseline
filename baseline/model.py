import torch
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
from transformers import BeamSearchScorer, LogitsProcessorList, NoBadWordsLogitsProcessor
from generation_bart import CopyConditionalGeneration
from constants import *

""" Model for text-to-text mapping using BART with copy mechanism for conditional generation

"""
class TextMappingModel(nn.Module):
    def __init__(self,
                 config,
                 vocabs):
        super().__init__()

        # vocabularies
        self.vocabs = vocabs

        # BERT encoder
        bert_config = config.bert_config
        bert_config.output_hidden_states = True
        self.bert_dim = bert_config.hidden_size
        self.extra_bert = config.extra_bert
        self.use_extra_bert = config.use_extra_bert
        if self.use_extra_bert:
            self.bert_dim *= 2
        self.bert_config = bert_config
        self.bert_dropout = nn.Dropout(p=config.bert_dropout)
        self.max_position_embeddings = config.max_position_embeddings
        self.num_beams = config.num_beams
        self.decoding_method = config.decoding_method
        self.SOT_weights = config.SOT_weights
        self.max_length = config.max_length
        self.use_copy = config.use_copy
        self._k = config.k

    def load_bert(self, name, cache_dir=None, tokenizer=None):
        """Load the pre-trained LM (used in training phrase)
        :param name (str): pre-trained LM name
        :param cache_dir (str): path to the LM cache directory
        """
        print('Loading pre-trained LM {}'.format(name))

        if self.use_copy:
            self.bert = CopyConditionalGeneration.from_pretrained(name, cache_dir=cache_dir,
                                                                         output_attentions=True)
            self.bert._k = self._k
        else:
            self.bert = AutoModelForSeq2SeqLM.from_pretrained(name, cache_dir=cache_dir)

    def forward(self, batch, decoder_input_ids=None, decoder_labels=None, decoder_masks=None, logger=None, tag=None,
                step=None, tokenizer=None):

        res = {}
        vocab_size = len(tokenizer)

        weight = torch.ones(vocab_size).to(batch.input_ids.device)
        self.bert._loss_weight = weight
        self.bert._vocab_size = vocab_size

        if self.use_copy:
            bart_outputs = self.encode(batch, decoder_input_ids=decoder_input_ids, decoder_labels=decoder_labels)
        else:
            bart_outputs = self.encode(batch, decoder_input_ids=decoder_input_ids)

        if decoder_labels is not None:

            if self.use_copy:
                weight[tokenizer.convert_tokens_to_ids(START_OF_TEMPLATE)] = self.SOT_weights
                loss = bart_outputs.loss
            else:
                weight[tokenizer.convert_tokens_to_ids(START_OF_TEMPLATE)] = self.SOT_weights
                loss = torch.nn.functional.cross_entropy(input=bart_outputs.logits.view(-1, vocab_size),
                                                         target=decoder_labels.view(-1), weight=weight)

            res['loss'] = loss

        return res

    def encode(self, batch, decoder_input_ids=None, decoder_labels=None, decoder_masks=None):
        '''
        Encode the input documents
        '''

        return self.bert(input_ids=batch.input_ids,
                         attention_mask=batch.attention_masks,
                         # 1 for tokens that are not masked, 0 for tokens that are masked.
                         decoder_input_ids=decoder_input_ids,
                         # For translation and summarization training, decoder_input_ids should be provided. If no decoder_input_ids is provided, the model will create this tensor by shifting the input_ids to the right for denoising pre-training following the paper.
                         labels=decoder_labels,
                         # decoder_attention_mask=decoder_masks, #Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
                         return_dict=True,
                         output_hidden_states=True,

                         )

    def generate(self, batch, num_beams, decoding_length, tokenizer, decoder_token_masks=None):
        '''
        From https://huggingface.co/transformers/main_classes/model.html?highlight=beamsearchscorer
        '''

        # list of bad words to prevent the model from predicting these outputs
        bad_words = [[tokenizer.sep_token_id], [tokenizer.pad_token_id]]
        logits_processor = NoBadWordsLogitsProcessor(bad_words_ids=bad_words, eos_token_id=tokenizer.eos_token_id)

        # seems that this is required if our model is a encoder-decoder architecture.
        model_kwargs = {
            "encoder_outputs": self.bert.get_encoder()(batch.input_ids.repeat_interleave(num_beams, dim=0),
                                                       batch.attention_masks.repeat_interleave(num_beams, dim=0),
                                                       return_dict=True),
        }
        # huggingface beamsearch workaround
        self.bert._cache_input_ids = batch.input_ids

        # create token for start decoding.
        decoder_input_ids = torch.ones((num_beams * batch.input_ids.size(0), 1), device=self.bert.device, dtype=torch.long)
        decoder_input_ids = decoder_input_ids * self.bert.config.decoder_start_token_id

        # decoder_input_ids = torch.tensor([self.bert.config.decoder_start_token_id] + batch.decoder_input_chunks[0][0][0], device=self.bert.device, dtype=torch.long)
        # decoder_input_ids = decoder_input_ids.repeat(num_beams, 1)

        if num_beams == 1:
            decoded_ids = self.bert.greedy_search(decoder_input_ids, max_length=decoding_length,
                                    logits_processor=logits_processor, **model_kwargs)
        else:
            beam_scorer = BeamSearchScorer(
            batch_size=batch.input_ids.size(0),
            max_length=decoding_length,
            num_beams=num_beams,
            device=self.bert.device,
            )
            decoded_ids = self.bert.beam_search(decoder_input_ids, beam_scorer, max_length=decoding_length,
                                                logits_processor=logits_processor, **model_kwargs)

        return decoded_ids

    def predict(self, batch, tokenizer, epoch=None):
        self.eval()

        with torch.no_grad():

            decoding_length = self.max_position_embeddings - 1
            if epoch is not None and epoch < 10:
                decoding_length = 10

            # Mask the tokens that are not present in the input document.
            # Only tokens in input document and the special tokens can be decoded.
            # Size: (batch, num_tokens)
            # TODO: Mask tokens that are invalid (i.e. not entities and not special tokens?)
            decoder_token_masks = torch.zeros(batch.input_ids.size(0), len(tokenizer), device=batch.input_ids.device,
                                              dtype=torch.bool)

            for batch_idx, input_ids in enumerate(batch.input_ids):
                decoder_token_masks[batch_idx, input_ids] = 1

            decoder_token_masks[:, tokenizer.pad_token_id] = 0
            decoder_token_masks[:, tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)] = 1
            decoder_token_masks[:, tokenizer.eos_token_id] = 1
            decoder_token_masks[:, tokenizer.bos_token_id] = 1


            decoded_ids = self.generate(batch, num_beams=1, decoding_length=decoding_length, tokenizer=tokenizer,
                                            decoder_token_masks=decoder_token_masks)

            res = {
                'decoded_ids': decoded_ids
            }

        self.train()
        return res

