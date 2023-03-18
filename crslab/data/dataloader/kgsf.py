# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/12/2
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

from copy import deepcopy

import torch
from loguru import logger
from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader
from crslab.data.dataloader.utils import add_start_end_token_idx, padded_tensor, get_onehot, truncate, merge_utt


class KGSFDataLoader(BaseDataLoader):
    """Dataloader for model KGSF.

    Notes:
        You can set the following parameters in config:

        - ``'context_truncate'``: the maximum length of context.
        - ``'response_truncate'``: the maximum length of response.
        - ``'entity_truncate'``: the maximum length of mentioned entities in context.
        - ``'word_truncate'``: the maximum length of mentioned words in context.

        The following values must be specified in ``vocab``:

        - ``'pad'``
        - ``'start'``
        - ``'end'``
        - ``'pad_entity'``
        - ``'pad_word'``

        the above values specify the id of needed special token.

        - ``'n_entity'``: the number of entities in the entity KG of dataset.

    """

    def __init__(self, opt, dataset, vocab):
        """

        Args:
            opt (Config or dict): config for dataloader or the whole system.
            dataset: data for model.
            vocab (dict): all kinds of useful size, idx and map between token and idx.

        """
        super().__init__(opt, dataset)
        self.n_entity = vocab['n_entity']
        self.pad_token_idx = vocab['pad']
        self.start_token_idx = vocab['start']
        self.end_token_idx = vocab['end']
        self.pad_entity_idx = vocab['pad_entity']
        self.pad_word_idx = vocab['pad_word']
        self.context_truncate = opt.get('context_truncate', None)
        self.response_truncate = opt.get('response_truncate', None)
        self.entity_truncate = opt.get('entity_truncate', None)
        self.word_truncate = opt.get('word_truncate', None)

    def get_pretrain_data(self, batch_size, shuffle=True):
        return self.get_data(self.pretrain_batchify, batch_size, shuffle, self.retain_recommender_target)

    def pretrain_batchify(self, batch):
        batch_context_entities = []
        batch_context_words = []
        for conv_dict in batch:
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(
                truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))

        return (padded_tensor(batch_context_words, self.pad_word_idx, pad_tail=False),
                get_onehot(batch_context_entities, self.n_entity))

    def rec_process_fn(self):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender':
                for movie in conv_dict['items']:
                    augment_conv_dict = deepcopy(conv_dict)
                    augment_conv_dict['item'] = movie
                    augment_dataset.append(augment_conv_dict)
        return augment_dataset

    def rec_batchify(self, batch):
        batch_context_entities = []
        batch_context_words = []
        batch_item = []
        for conv_dict in batch:
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(
                truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))
            batch_item.append(conv_dict['item'])

        return (padded_tensor(batch_context_entities, self.pad_entity_idx, pad_tail=False),
                padded_tensor(batch_context_words,
                              self.pad_word_idx, pad_tail=False),
                get_onehot(batch_context_entities, self.n_entity),
                torch.tensor(batch_item, dtype=torch.long))

    def rec_interact(self, data, stage):
        # pack: context_entities, context_words, entities, movie = batch
        batch_context_entities = []
        batch_context_words = []
        batch_item = []
        
        for conv_dict in data:
            logger.info("conv_dict")
            logger.info(conv_dict)
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(
                truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))

            # we do not have the key 'item' in the data so we append dummy id or previous recommendation id to prevent crashing.
            if len(conv_dict['interaction_history']) <= 0:
                dummy_itemid = 0
                batch_item.append(dummy_itemid)
            else:
                batch_item.append(conv_dict['interaction_history'][-1])
                
        return (padded_tensor(batch_context_entities, self.pad_entity_idx, pad_tail=False),
                padded_tensor(batch_context_words,
                              self.pad_word_idx, pad_tail=False),
                get_onehot(batch_context_entities, self.n_entity),
                torch.tensor(batch_item, dtype=torch.long))

    def _process_rec_context(self, context_tokens):
        compact_context = []
        for i, utterance in enumerate(context_tokens):
            if i != 0:
                utterance.insert(0, self.sent_split_idx)
            compact_context.append(utterance)
        compat_context = truncate(merge_utt(compact_context),
                                  self.context_truncate - 2,
                                  truncate_tail=False)
        compat_context = add_start_end_token_idx(compat_context,
                                                 self.start_token_idx,
                                                 self.end_token_idx)
        return compat_context

    def _process_history(self, context_items, item_id=None):
        input_ids = truncate(context_items,
                             max_length=self.context_truncate,
                             truncate_tail=False)
        input_mask = [1] * len(input_ids)
        sample_negs = []
        seq_set = set(input_ids)
        for _ in input_ids:
            sample_negs.append(self._neg_sample(seq_set))

        if item_id is not None:
            target_pos = input_ids[1:] + [item_id]
            return input_ids, target_pos, input_mask, sample_negs
        else:
            return input_ids, input_mask, sample_negs

    def conv_process_fn(self, *args, **kwargs):
        return self.retain_recommender_target()

    def conv_batchify(self, batch):
        batch_context_tokens = []
        batch_context_entities = []
        batch_context_words = []
        batch_response = []
        
        for conv_dict in batch:
            batch_context_tokens.append(
                truncate(merge_utt(conv_dict['context_tokens']), self.context_truncate, truncate_tail=False))
            batch_context_entities.append(
                truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            batch_context_words.append(
                truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))
            batch_response.append(
                add_start_end_token_idx(truncate(conv_dict['response'], self.response_truncate - 2),
                                        start_token_idx=self.start_token_idx,
                                        end_token_idx=self.end_token_idx))

        return (padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
                padded_tensor(batch_context_entities,
                              self.pad_entity_idx, pad_tail=False),
                padded_tensor(batch_context_words,
                              self.pad_word_idx, pad_tail=False),
                padded_tensor(batch_response, self.pad_token_idx))

    def conv_interact(self, batch):
        
        batch_context_tokens = []
        batch_context_entities = []
        batch_context_words = []
        batch_response = []
        
        for conv_dict in batch:
            for x in conv_dict['context_tokens']:   # x is a list
                for y in x:     # y is int
                    batch_context_tokens.append(y)
            for item in conv_dict['context_entities']:
                batch_context_entities.append(item)
            for item in conv_dict['context_words']:
                batch_context_words.append(item)
            for item in conv_dict['context_tokens'][-1]:
                batch_response.append(item)
            # batch_context_tokens.append(
            #     truncate(merge_utt(conv_dict['context_tokens']), self.context_truncate, truncate_tail=False))
            # batch_context_entities.append(
            #     truncate(conv_dict['context_entities'], self.entity_truncate, truncate_tail=False))
            # batch_context_words.append(
            #     truncate(conv_dict['context_words'], self.word_truncate, truncate_tail=False))
            
            # we do not have the key 'response' in the data so we take the last context tokens sentence as response 
            # batch_response.append(
            #     add_start_end_token_idx(truncate(conv_dict['context_tokens'][-1], self.response_truncate - 2),
            #                             start_token_idx=self.start_token_idx,
            #                             end_token_idx=self.end_token_idx))
            
            # batch_response.append(
            #    add_start_end_token_idx(truncate(conv_dict['response'], self.response_truncate - 2),
                                    #    start_token_idx=self.start_token_idx,
            #                            end_token_idx=self.end_token_idx))
            
            # dummy_response = [43, 142, 19, 23, 28, 6, 47, 21, 10]
            # batch_response.append(
            #     add_start_end_token_idx(truncate(dummy_response, self.response_truncate - 2),
            #                             start_token_idx=self.start_token_idx,
            #                             end_token_idx=self.end_token_idx))

        # return (padded_tensor(batch_context_tokens, self.pad_token_idx, pad_tail=False),
        #         padded_tensor(batch_context_entities,
        #                       self.pad_entity_idx, pad_tail=False),
        #         padded_tensor(batch_context_words,
        #                       self.pad_word_idx, pad_tail=False),
        #         padded_tensor(batch_response, self.pad_token_idx))
        
        return (
            torch.tensor([batch_context_tokens]),
            torch.tensor([batch_context_entities]),
            torch.tensor([batch_context_words]),
            torch.tensor([batch_response]),
        )
        
        
    def policy_batchify(self, *args, **kwargs):
        pass
