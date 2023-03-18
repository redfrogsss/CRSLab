# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2021/1/3
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os

import torch
from loguru import logger

from crslab.data import get_dataloader, dataset_language_map
from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt

import re


class KGSFSystem(BaseSystem):
    """This is the system for KGSF model"""

    def __init__(self, opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data, restore_system=False,
                 interact=False, debug=False, tensorboard=False):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.
            tensorboard (bool, optional) Indicating if we monitor the training performance in tensorboard. Defaults to False. 

        """
        super(KGSFSystem, self).__init__(opt, train_dataloader, valid_dataloader, test_dataloader, vocab, side_data,
                                         restore_system, interact, debug, tensorboard)

        self.ind2tok = vocab['ind2tok']
        self.end_token_idx = vocab['end']
        self.item_ids = side_data['item_entity_ids']

        self.pretrain_optim_opt = self.opt['pretrain']
        self.rec_optim_opt = self.opt['rec']
        self.conv_optim_opt = self.opt['conv']
        self.pretrain_epoch = self.pretrain_optim_opt['epoch']
        self.rec_epoch = self.rec_optim_opt['epoch']
        self.conv_epoch = self.conv_optim_opt['epoch']
        self.pretrain_batch_size = self.pretrain_optim_opt['batch_size']
        self.rec_batch_size = self.rec_optim_opt['batch_size']
        self.conv_batch_size = self.conv_optim_opt['batch_size']
        
        self.language = dataset_language_map[self.opt['dataset']]

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, item in zip(rec_ranks, item_label):
            item = self.item_ids.index(item)
            self.evaluator.rec_evaluate(rec_rank, item)

    
    def rec_generate(self, rec_predict, item_label):
        result = None
        logger.info("rec_predict")
        logger.info(rec_predict)
        rec_predict = rec_predict.cpu()

        logger.info("rec_predict cpu")
        logger.info(rec_predict)
        rec_predict = rec_predict[:, self.item_ids]

        logger.info("rec_predict sliced list")
        logger.info(rec_predict)
        
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
    
        logger.info("rec_ranks")
        logger.info(rec_ranks)
        
        logger.info("item_label")
        logger.info(item_label)
    
        result = rec_ranks[0][0]
        
        # Display id of the recommend item
        logger.info("rec_generate Recommended item: ")
        logger.info(result)

        return result
    
    def conv_evaluate(self, prediction, response):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])
            
            
    def conv_generate(self, prediction, response):
        prediction = prediction.tolist()
        return ind2txt(prediction[0], self.ind2tok, self.end_token_idx)

    def step(self, batch, stage, mode):
        batch = [ele.to(self.device) for ele in batch]
        logger.info("batch")
        logger.info(batch)
        if stage == 'pretrain':
            info_loss = self.model.forward(batch, stage, mode)
            if info_loss is not None:
                self.backward(info_loss.sum())
                info_loss = info_loss.sum().item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == 'rec':
            rec_loss, info_loss, rec_predict = self.model.forward(batch, stage, mode)
            if info_loss:
                loss = rec_loss + 0.025 * info_loss
            else:
                loss = rec_loss
            if mode == "train":
                self.backward(loss.sum())
            else:
                if mode == "generate":
                    return self.rec_generate(rec_predict, batch[-1])
                else:
                    self.rec_evaluate(rec_predict, batch[-1])
            rec_loss = rec_loss.sum().item()
            if mode != "generate":
                self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
                if info_loss:
                    info_loss = info_loss.sum().item()
                    self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == "conv":
            if mode != "test":
                gen_loss, pred = self.model.forward(batch, stage, mode)
                if mode == 'train':
                    self.backward(gen_loss.sum())
                else:
                    if mode == 'generate':
                        return self.conv_generate(pred, batch[-1])
                    else:
                        self.conv_evaluate(pred, batch[-1])
                gen_loss = gen_loss.sum().item()
                self.evaluator.optim_metrics.add("gen_loss", AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                pred = self.model.forward(batch, stage, mode)
                self.conv_evaluate(pred, batch[-1])
        else:
            raise

    def pretrain(self):
        self.init_optim(self.pretrain_optim_opt, self.model.parameters())

        for epoch in range(self.pretrain_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Pretrain epoch {str(epoch)}]')
            for batch in self.train_dataloader.get_pretrain_data(self.pretrain_batch_size, shuffle=False):
                self.step(batch, stage="pretrain", mode='train')
            self.evaluator.report()

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.model.parameters())

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Recommendation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                    self.step(batch, stage='rec', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
                # early stop
                metric = self.evaluator.rec_metrics['hit@1'] + self.evaluator.rec_metrics['hit@50']
                if self.early_stop(metric):
                    break
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(self.rec_batch_size, shuffle=False):
                self.step(batch, stage='rec', mode='test')
            self.evaluator.report(mode='test')

    def train_conversation(self):
        # if os.environ["cuda_visible_devices"] == '-1':
        #     self.model.freeze_parameters()
        # else:
        #     self.model.module.freeze_parameters()
        self.model.freeze_parameters()
        self.init_optim(self.conv_optim_opt, self.model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f'[Conversation epoch {str(epoch)}]')
            logger.info('[Train]')
            for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='train')
            self.evaluator.report(epoch=epoch, mode='train')
            # val
            logger.info('[Valid]')
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                    self.step(batch, stage='conv', mode='val')
                self.evaluator.report(epoch=epoch, mode='val')
        # test
        logger.info('[Test]')
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
                self.step(batch, stage='conv', mode='test')
            self.evaluator.report(mode='test')

    def fit(self):
        self.pretrain()
        self.train_recommender()
        self.train_conversation()

    def interact(self):
        logger.info('[Interact]')
        self.init_interact()
        
        input_text = self.get_input(self.language)
        while not self.finished:
            # rec
            if hasattr(self, 'model'):
                rec_input = self.process_input(input_text, 'rec')
                
                logger.info("rec_input")
                logger.info(rec_input)
                
                
                logger.info("Generating Recommending Items...")
                rec_loss, info_loss, rec_predict = self.model.forward(rec_input, "rec", "generate")
                scores = rec_predict.cpu()[0]
                
                # logger.info("scores")
                # logger.info(scores)
                
                k = 100
                _, rank = torch.topk(scores, k, dim=-1)
                
                item_ids = rank.tolist()
                # logger.info("rank.tolist()")
                # logger.info(item_ids)
                rec_id = item_ids[0]  # get the first item from rank
                
                # search for the first non-duplicated item
                for i, vi in enumerate(item_ids):
                    for j, vj in enumerate(self.context["rec"]['interaction_history']):
                        if item_ids[i] == self.context["rec"]['interaction_history'][j]:
                            continue
                        rec_id = item_ids[i]  # get the first item from rank
                        break
                        
                self.context["rec"]["interaction_history"].append(rec_id)
                # logger.info(self.context["rec"]["interaction_history"])
                        
                logger.info("rec_id")
                logger.info(rec_id)
                
                rec_name = self.convert_id_to_movie_entity(rec_id)
                logger.info("rec_name")
                logger.info(rec_name)
                
                rec_name = self.recommendation_postprocessing(rec_name)
                logger.info("rec_postprocessing")
                logger.info(rec_name)
                
                self.send_response_to_frontend(rec_name, False)
                
                self.update_context(stage='rec', item_ids=[rec_id], entity_ids=[rec_id])
                
                logger.info("End of rec")
                
                conv_input = self.process_input(input_text, 'conv')
                
                logger.info("conv_input")
                logger.info(conv_input)
                
                logger.info("Generating Conversation...")
                # loss, pred = self.model.forward(conv_input, 'conv', 'generate')[0]
                pred = self.model.forward(conv_input, 'conv', 'test')[0]
                # prediction = pred.tolist()
                
                # logger.info(prediction)
                conv_result = ind2txt(pred, self.ind2tok, self.end_token_idx)
                
                logger.info("conv_result")
                logger.info(conv_result)
                
                conv_result = self.response_postprocessing(conv_result)
                logger.info("conv_result postprocessed")
                logger.info(conv_result)
                
                # exit()
                
                self.send_response_to_frontend(conv_result, True, res_type="recommend")
                
                conv_result = conv_result.replace("[Response]", "")
                
                logger.info("Updating context...")
                token_ids, entity_ids, movie_ids, word_ids = self.convert_to_id(conv_result, "conv")
                self.update_context(stage="conv", word_ids=word_ids, token_ids=token_ids)
                
            else:
                logger.info("no attr 'rec_model' or 'model'")

            input_text = self.get_input(self.language)

    def process_input(self, input_text, stage):
        token_ids, entity_ids, movie_ids, word_ids = self.convert_to_id(input_text, stage)
        self.update_context(stage, token_ids, entity_ids, movie_ids, word_ids)

        data = [{'role': 'Seeker',
                 'context_tokens': self.context[stage]['context_tokens'],
                 'context_entities': self.context[stage]['context_entities'],
                 'context_words': self.context[stage]['context_words'],
                 'context_items': self.context[stage]['context_items'],
                 'user_profile': self.context[stage]['user_profile'],
                 'interaction_history': self.context[stage]['interaction_history']
                 }]
        
        # hardcode data for testing
        # data = [{'role': 'Recommender', 'context_tokens': [[46], [167, 86, 28, 403]], 'response': [129, 5, 88, 28, 6, 130, 434, 10], 'context_entities': [], 'context_words': [13136, 13848, 23130], 'context_items': [], 'items': [], 'interaction_history': []}]

        dataloader = get_dataloader(self.opt, data, self.vocab)
    
        logger.info("process_input stage: " + stage)
        
        if stage == 'rec':
            data = dataloader.rec_interact(data, stage)
        elif stage == 'conv':
            data = dataloader.conv_interact(data)
            
        # use training data for testing model output
        # if stage == 'conv':
        #     for batch in self.train_dataloader.get_conv_data(batch_size=1, shuffle=False):
        #         logger.info("training data")
        #         logger.info(batch)
        #         data = batch
        #         break
        #     # exit()
        #     return data

        logger.info("process_input() data result")
        self.vocab["id2word"] = {v: k for k, v in self.vocab["word2id"].items()}    # create vocab to convert id to word
        
        logger.info("self.context[stage]['context_tokens']")
        logger.info([self.vocab["ind2tok"][i] for i in self.context[stage]['context_tokens'][-1]])
        
        if(stage == "rec"):
            logger.info(data[1].tolist()[0])
            logger.info("data[1]--context_words")
            logger.info([self.vocab["id2word"][i] for i in data[1].tolist()[0]])
            
        if stage == "conv":
            logger.info("data")
            logger.info(data)
            
            if data[2].tolist()[0][0] != 0:
                logger.info("data[2]--context_words")
                logger.info([self.vocab["id2word"][i] for i in data[2].tolist()[0]])
            
        # check data in batchify
        # for batch in self.train_dataloader.get_conv_data(batch_size=self.conv_batch_size, shuffle=False):
            # logger.info("obtaining training data, will not reach this line")
            # dataloader.conv_batchify(batch)
        # exit()
        
        # data = [[13136, 13848, 23130]]
        # for v in data:
            # logger.info(v)
            # logger.info([self.vocab["ind2tok"][i] for i in v])
            # logger.info([self.vocab["id2word"][i] for i in v])
        # exit()
        
        
        data = [ele.to(self.device) if isinstance(ele, torch.Tensor) else ele for ele in data]
        
        # logger.info(data)
        return data

    def convert_to_id(self, text, stage):
        if self.language == 'zh':
            tokens = self.tokenize(text, 'pkuseg')
        elif self.language == 'en':
            tokens = self.tokenize(text, 'nltk')
        else:
            raise
        
        # logger.info("entity length")
        # logger.info(len(self.side_data['entity_kg']['entity']))  # return 59809
        
        # logger.info("word length")
        # logger.info(len(self.side_data['word_kg']['entity']))    # return 23713
        
        # logger.info(self.side_data['entity_kg']['entity'])
        
        # logger.info("self.vocab['entity2id']")
        # logger.info(list(self.vocab['entity2id'].keys()))
        
        # logger.info("ind2tok")
        # logger.info(self.vocab["ind2tok"])
        
        # exit()
        
        logger.info("Finding entities from the input text using fuzzy logic")
        # entities = self.link(tokens, self.side_data['entity_kg']['entity'])
        # entities = self.link(tokens, self.side_data['entity_kg']['entity'], "entity")
        entities = self.link(tokens, list(self.vocab['entity2id'].keys()), "entity")
        logger.info("entities")
        logger.info(entities)
        
        logger.info("Finding words from the input text using fuzzy logic")
        # words = self.link(tokens, self.side_data['word_kg']['entity'])
        # words = self.link(tokens, self.side_data['word_kg']['entity'], "word")
        words = self.link(tokens, list(self.vocab['word2id'].keys()), "word")
        # words = self.link(tokens, list(self.vocab['tok2ind'].keys()), "word")
        logger.info("words")
        logger.info(words)

        # logger.info(self.opt['tokenize'])
        if self.opt['tokenize'] in ('gpt2', 'bert'):
            language = dataset_language_map[self.opt['dataset']]
            path = os.path.join(PRETRAIN_PATH, self.opt['tokenize'][stage], language)
            tokens = self.tokenize(text, 'bert', path)

        # logger.info(self.vocab.keys())
        token_ids = [self.vocab['tok2ind'].get(token, self.vocab['unk']) for token in tokens]
        entity_ids = [self.vocab['entity2id'][entity] for entity in entities if
                      entity in self.vocab['entity2id']]
        logger.info("entity_ids")
        logger.info(entity_ids)
        movie_ids = [entity_id for entity_id in entity_ids if entity_id in self.item_ids]
        word_ids = [self.vocab['word2id'][word] for word in words if word in self.vocab['word2id']]
        

        return token_ids, entity_ids, movie_ids, word_ids
    
    def convert_id_to_movie_entity(self, movie_id):
        result = self.vocab['id2entity'][movie_id]
        return result

    def recommendation_postprocessing(self, recommendation):
        # input: <http://dbpedia.org/resource/A_Shot_in_the_Dark_(1964_film)>
        # expected output: [Recommendation] A Shot in the Dark (1964_film)
        recommendation = recommendation[1:]     # remove first character
        recommendation = recommendation[:-1]    # remove last character
        recommendation = recommendation.replace("http://dbpedia.org/resource/", "")
        recommendation = recommendation.replace("_", " ")
        recommendation = f"[Recommend] {recommendation}"
        return recommendation
    
    def response_postprocessing(self, response):
        # input: __start__ I have not forward something reason to @123456  
        # expected output: [Response] I have not forward something reason to this
        response = response.replace("__start__ __start__", "[Response] ")
        
        response = re.sub(r'@\d+', "this", response)    # replace @123456 in the response
        
        return response 