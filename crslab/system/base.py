# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2021/1/9
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

# UPDATE:
# @Time   : 2021/11/5
# @Author : Zhipeng Zhao
# @Email  : oran_official@outlook.com

import json
import os
from abc import ABC, abstractmethod
import numpy as np
import random
import nltk
import torch
from fuzzywuzzy.process import extractOne
from loguru import logger
from nltk import word_tokenize
from torch import optim
from transformers import AdamW, Adafactor

from crslab.config import SAVE_PATH
from crslab.evaluator import get_evaluator
from crslab.evaluator.metrics.base import AverageMetric
from crslab.model import get_model
from crslab.system.utils import lr_scheduler
from crslab.system.utils.functions import compute_grad_norm

import time
import requests

optim_class = {}
optim_class.update({k: v for k, v in optim.__dict__.items() if not k.startswith('__') and k[0].isupper()})
optim_class.update({'AdamW': AdamW, 'Adafactor': Adafactor})
lr_scheduler_class = {k: v for k, v in lr_scheduler.__dict__.items() if not k.startswith('__') and k[0].isupper()}
transformers_tokenizer = ('bert', 'gpt2')


class BaseSystem(ABC):
    """Base class for all system"""

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
        self.opt = opt
        if opt["gpu"] == [-1]:
            self.device = torch.device('cpu')
        elif len(opt["gpu"]) == 1:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cuda')
        # seed
        if 'seed' in opt:
            seed = int(opt['seed'])
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            logger.info(f'[Set seed] {seed}')
        # data
        if debug:
            self.train_dataloader = valid_dataloader
            self.valid_dataloader = valid_dataloader
            self.test_dataloader = test_dataloader
        else:
            self.train_dataloader = train_dataloader
            self.valid_dataloader = valid_dataloader
            self.test_dataloader = test_dataloader
        self.vocab = vocab
        self.side_data = side_data
        # model
        if 'model' in opt:
            self.model = get_model(opt, opt['model'], self.device, vocab, side_data).to(self.device)
        else:
            if 'rec_model' in opt:
                self.rec_model = get_model(opt, opt['rec_model'], self.device, vocab['rec'], side_data['rec']).to(
                    self.device)
            if 'conv_model' in opt:
                self.conv_model = get_model(opt, opt['conv_model'], self.device, vocab['conv'], side_data['conv']).to(
                    self.device)
            if 'policy_model' in opt:
                self.policy_model = get_model(opt, opt['policy_model'], self.device, vocab['policy'],
                                              side_data['policy']).to(self.device)
        model_file_name = opt.get('model_file', f'{opt["model_name"]}.pth')
        self.model_file = os.path.join(SAVE_PATH, model_file_name)
        if restore_system:
            self.restore_model()

        if not interact:
            self.evaluator = get_evaluator(opt.get('evaluator', 'standard'), opt['dataset'], tensorboard)

    def init_optim(self, opt, parameters):
        self.optim_opt = opt
        parameters = list(parameters)
        if isinstance(parameters[0], dict):
            for i, d in enumerate(parameters):
                parameters[i]['params'] = list(d['params'])

        # gradient acumulation
        self.update_freq = opt.get('update_freq', 1)
        self._number_grad_accum = 0

        self.gradient_clip = opt.get('gradient_clip', -1)

        self.build_optimizer(parameters)
        self.build_lr_scheduler()

        if isinstance(parameters[0], dict):
            self.parameters = []
            for d in parameters:
                self.parameters.extend(d['params'])
        else:
            self.parameters = parameters

        # early stop
        self.need_early_stop = self.optim_opt.get('early_stop', False)
        if self.need_early_stop:
            logger.debug('[Enable early stop]')
            self.reset_early_stop_state()

    def build_optimizer(self, parameters):
        optimizer_opt = self.optim_opt['optimizer']
        optimizer = optimizer_opt.pop('name')
        self.optimizer = optim_class[optimizer](parameters, **optimizer_opt)
        logger.info(f"[Build optimizer: {optimizer}]")

    def build_lr_scheduler(self):
        """
        Create the learning rate scheduler, and assign it to self.scheduler. This
        scheduler will be updated upon a call to receive_metrics. May also create
        self.warmup_scheduler, if appropriate.

        :param state_dict states: Possible state_dict provided by model
            checkpoint, for restoring LR state
        :param bool hard_reset: If true, the LR scheduler should ignore the
            state dictionary.
        """
        if self.optim_opt.get('lr_scheduler', None):
            lr_scheduler_opt = self.optim_opt['lr_scheduler']
            lr_scheduler = lr_scheduler_opt.pop('name')
            self.scheduler = lr_scheduler_class[lr_scheduler](self.optimizer, **lr_scheduler_opt)
            logger.info(f"[Build scheduler {lr_scheduler}]")

    def reset_early_stop_state(self):
        self.best_valid = None
        self.drop_cnt = 0
        self.impatience = self.optim_opt.get('impatience', 3)
        if self.optim_opt['stop_mode'] == 'max':
            self.stop_mode = 1
        elif self.optim_opt['stop_mode'] == 'min':
            self.stop_mode = -1
        else:
            raise
        logger.debug('[Reset early stop state]')

    @abstractmethod
    def fit(self):
        """fit the whole system"""
        pass

    @abstractmethod
    def step(self, batch, stage, mode):
        """calculate loss and prediction for batch data under certrain stage and mode

        Args:
            batch (dict or tuple): batch data
            stage (str): recommendation/policy/conversation etc.
            mode (str): train/valid/test
        """
        pass

    def backward(self, loss):
        """empty grad, backward loss and update params

        Args:
            loss (torch.Tensor):
        """
        self._zero_grad()

        if self.update_freq > 1:
            self._number_grad_accum = (self._number_grad_accum + 1) % self.update_freq
            loss /= self.update_freq
        loss.backward(loss.clone().detach())

        self._update_params()

    def _zero_grad(self):
        if self._number_grad_accum != 0:
            # if we're accumulating gradients, don't actually zero things out yet.
            return
        self.optimizer.zero_grad()

    def _update_params(self):
        if self.update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            # self._number_grad_accum is updated in backward function
            if self._number_grad_accum != 0:
                return

        if self.gradient_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.parameters, self.gradient_clip
            )
            self.evaluator.optim_metrics.add('grad norm', AverageMetric(grad_norm))
            self.evaluator.optim_metrics.add(
                'grad clip ratio',
                AverageMetric(float(grad_norm > self.gradient_clip)),
            )
        else:
            grad_norm = compute_grad_norm(self.parameters)
            self.evaluator.optim_metrics.add('grad norm', AverageMetric(grad_norm))

        self.optimizer.step()

        if hasattr(self, 'scheduler'):
            self.scheduler.train_step()

    def adjust_lr(self, metric=None):
        """adjust learning rate w/o metric by scheduler

        Args:
            metric (optional): Defaults to None.
        """
        if not hasattr(self, 'scheduler') or self.scheduler is None:
            return
        self.scheduler.valid_step(metric)
        logger.debug('[Adjust learning rate after valid epoch]')

    def early_stop(self, metric):
        if not self.need_early_stop:
            return False
        if self.best_valid is None or metric * self.stop_mode > self.best_valid * self.stop_mode:
            self.best_valid = metric
            self.drop_cnt = 0
            logger.info('[Get new best model]')
            return False
        else:
            self.drop_cnt += 1
            if self.drop_cnt >= self.impatience:
                logger.info('[Early stop]')
                return True

    def save_model(self):
        r"""Store the model parameters."""
        state = {}
        if hasattr(self, 'model'):
            state['model_state_dict'] = self.model.state_dict()
        if hasattr(self, 'rec_model'):
            state['rec_state_dict'] = self.rec_model.state_dict()
        if hasattr(self, 'conv_model'):
            state['conv_state_dict'] = self.conv_model.state_dict()
        if hasattr(self, 'policy_model'):
            state['policy_state_dict'] = self.policy_model.state_dict()

        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save(state, self.model_file)
        logger.info(f'[Save model into {self.model_file}]')

    def restore_model(self):
        r"""Store the model parameters."""
        if not os.path.exists(self.model_file):
            raise ValueError(f'Saved model [{self.model_file}] does not exist')
        checkpoint = torch.load(self.model_file, map_location=self.device)
        if hasattr(self, 'model'):
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("'model' attr loaded")
        if hasattr(self, 'rec_model'):
            self.rec_model.load_state_dict(checkpoint['rec_state_dict'])
            logger.info("'rec_model' attr loaded")
        if hasattr(self, 'conv_model'):
            self.conv_model.load_state_dict(checkpoint['conv_state_dict'])
            logger.info("'conv_model' attr loaded")
        if hasattr(self, 'policy_model'):
            self.policy_model.load_state_dict(checkpoint['policy_state_dict'])
            logger.info("'policy_model' attr loaded")
        logger.info(f'[Restore model from {self.model_file}]')

    @abstractmethod
    def interact(self):
        pass

    def init_interact(self):
        self.finished = False
        self.context = {
            'rec': {},
            'conv': {}
        }
        for key in self.context:
            self.context[key]['context_tokens'] = []
            self.context[key]['context_entities'] = []
            self.context[key]['context_words'] = []
            self.context[key]['context_items'] = []
            self.context[key]['user_profile'] = []
            self.context[key]['interaction_history'] = []
            self.context[key]['entity_set'] = set()
            self.context[key]['word_set'] = set()
            
    def send_context_to_backend(self, chat_id, model, stage, context):
        
        context_data = {
            "chat_id": chat_id,
            "model": model,
            "stage": stage,
            "context_tokens": json.dumps(context["context_tokens"]),
            "context_entities": json.dumps(context["context_entities"]),
            "context_words": json.dumps(context["context_words"]),
            "context_items": json.dumps(context["context_items"]),
            "user_profile": json.dumps(context["user_profile"]),
            "interaction_history": json.dumps(context["interaction_history"]),
        }
        
        # if backend has no context, create a new one
        response = requests.get(self.getBackendUrl() + "/context?chat_id=" + chat_id + "&model=" + model + "&stage=" + stage)
        data = response.json()
        if data["result"] == "None":
            logger.info("send_context_to_backned(): no context found. Creating new one.")
            response = requests.post(self.getBackendUrl() + "/context", context_data)
            logger.info(response.json())
        else:
            logger.info("send_context_to_backned(): context updated.")
            response = requests.put(self.getBackendUrl() + "/context", context_data)
            logger.info(response.json())
    
    def get_context_from_backend(self, chat_id, model, stage):
        response = requests.get(self.getBackendUrl() + "/context?chat_id=" + chat_id + "&model=" + model + "&stage=" + stage)
        data = response.json()
        context = data["result"]
        return context

    def update_context(self, stage, chat_id=None, token_ids=None, entity_ids=None, item_ids=None, word_ids=None, model="kgsf"):
        if chat_id is None:
            logger.info("update_context(): chat_id is None")
            # return None
        
        if token_ids is not None:
            self.context[stage]['context_tokens'].append(token_ids)
        if item_ids is not None:
            self.context[stage]['context_items'] += item_ids
        if entity_ids is not None:
            for entity_id in entity_ids:
                if entity_id not in self.context[stage]['entity_set']:
                    self.context[stage]['entity_set'].add(entity_id)
                    self.context[stage]['context_entities'].append(entity_id)
        if word_ids is not None:
            for word_id in word_ids:
                 if word_id not in self.context[stage]['word_set']:
                     self.context[stage]['word_set'].add(word_id)
                     self.context[stage]['context_words'].append(word_id)
        
        # send context to backend
        logger.info(chat_id)
        logger.info(model)
        logger.info(self.context[stage])
        self.send_context_to_backend(chat_id, model, stage, self.context[stage])
        
    def get_input(self, language):
        logger.info(f"Waiting for new {language} message...")

        # if language == 'zh':
        #     language = 'chinese'
        # elif language == 'en':
        #     language = 'english'
        # else:
        #     raise
        if language == 'zh':
            language = 'ZH'
        elif language == 'en':
            language = 'EN'
        else:
            raise
        # text = input(f"Enter Your Message in {language}: ")

        text, chat_id = self.get_input_from_frontend(language)

        if '[EXIT]' in text:
            self.finished = True
        return text, chat_id

    def tokenize(self, text, tokenizer, path=None):
        tokenize_fun = getattr(self, tokenizer + '_tokenize')
        if path is not None:
            return tokenize_fun(text, path)
        else:
            return tokenize_fun(text)

    def nltk_tokenize(self, text):
        nltk.download('punkt')
        return word_tokenize(text)

    def bert_tokenize(self, text, path):
        if not hasattr(self, 'bert_tokenizer'):
            from transformers import AutoTokenizer
            self.bert_tokenizer = AutoTokenizer.from_pretrained(path)
        return self.bert_tokenizer.tokenize(text)

    def gpt2_tokenize(self, text, path):
        if not hasattr(self, 'gpt2_tokenizer'):
            from transformers import AutoTokenizer
            self.gpt2_tokenizer = AutoTokenizer.from_pretrained(path)
        return self.gpt2_tokenizer.tokenize(text)

    def pkuseg_tokenize(self, text):
        if not hasattr(self, 'pkuseg_tokenizer'):
            #import pkuseg
            import spacy_pkuseg as pkuseg
            self.pkuseg_tokenizer = pkuseg.pkuseg()
        return self.pkuseg_tokenizer.cut(text)
    

#     def link(self, tokens, entities):
#         linked_entities = []
#         for token in tokens:
#             entity = extractOne(token, entities, score_cutoff=90)
#             if entity:
#                 linked_entities.append(entity[0])
#         return linked_entities
    
    def link(self, tokens, entities, entities_type = "entity"):
        # entities_type == "word" || "entity"
        linked_entities = []
        for token in tokens:
            entity = self.getExtractResult(token = token, token_type=entities_type)
            # logger.info("token : getExtractResult(token)")
            # logger.info(f"{token} : {entity}")
            if entity != None:
                if len(entity) > 0:
                    linked_entities.append(entity)
            else:
                entity = extractOne(token, entities, score_cutoff=90)
                if entity:
                    linked_entities.append(entity[0])
                    entity = entity[0]
                
                self.saveExtractResult(token=token, word=entity, token_type=entities_type)
        return linked_entities
    
    
    def getExtractResult(self, token, token_type):
        if token_type == "entity":
            endpoint = "/extractEntity"
        else:
            endpoint = "/extractWord"
        response = requests.get(self.getBackendUrl() + endpoint + "?token=" + token)
        data = response.json()
        # logger.info("data")
        # logger.info(data)
        result = data["result"]
        if result == "None":
            return None
        else:
            result = str(data["result"]["word"])
            if result == "NULL":
                result = ""
            if len(result) == 0:
                result = ""
            return result
        
    def saveExtractResult(self, token, word, token_type):
        if word == None:
            word = ""
        logger.info(f"Save Extract Result for '{token} : {word}'.")
        if token_type == "entity":
            endpoint = "/extractEntity"
        else:
            endpoint = "/extractWord"
        requests.post(self.getBackendUrl() + endpoint + "?token=" + str(token) + "&word=" + str(word))
    
    def get_input_from_frontend(self, language):
        result = "None"

        while result == "None" or res_language != language:
            # response = requests.get("http://192.168.0.37:3001/input_queue")
            response = requests.get(self.getBackendUrl() + "/input_queue")
            data = response.json()
            result = data["result"]
            if result != "None":
                res_language = str(data["result"]["language"])
            time.sleep(1)

        message = str(data["result"]["message"])
        chat_id = str(data["result"]["chat_id"])
        
        logger.info("message")
        logger.info(message)
        logger.info("chat_id")
        logger.info(chat_id)
        
        res_language = str(data["result"]["language"])
        
        return message, chat_id
        
    def getBackendUrl(self):
        return "http://192.168.0.37:3001/api"

    def send_response_to_frontend(self, content="None", done=False, res_type="text"):

        # response = requests.get("http://192.168.0.30:3001/input_queue")
        response = requests.get(self.getBackendUrl() + "/input_queue")
        data = response.json()
        chat_id = str(data["result"]["chat_id"])
        user_id = str(data["result"]["user_id"])

        if content == "None":
            logger.info("send_response_to_frontend(): no content found.")

        req_obj = {
            "content": content,
            "type": res_type,
            "user_id": user_id,
            "chat_id": chat_id,
        }

        requests.post(self.getBackendUrl() + "/message?content=" + content +
                      "&type=" + res_type + "&user_id=" + user_id + "&chat_id=" + chat_id)

        if done == True:
            requests.put(self.getBackendUrl() + "/input_queue")

    def get_context_data(self, model, stage, chat_id=None):
        
        logger.info("get_context_data()")
        
        if chat_id is None:
            logger.info("chat_id is None")
            return
        
        self.context[stage]['context_tokens'] = []
        self.context[stage]['context_entities'] = []
        self.context[stage]['context_words'] = []
        self.context[stage]['context_items'] = []
        self.context[stage]['user_profile'] = []
        self.context[stage]['interaction_history'] = []
        # self.context[stage]['entity_set'] = set()
        # self.context[stage]['word_set'] = set()
        
        # get input from backend then assign value to self.context[stage]
        context_data = self.get_context_from_backend(chat_id, model, stage)
        
        if context_data is None or context_data == "None":
            logger.info("context_data is None")
            return
        
        # logger.info(json.loads(context_data['context_entities']))
        # logger.info(type(json.loads(context_data['context_entities'])))
        
        self.context[stage]['context_entities'] = json.loads(context_data['context_entities'])
        self.context[stage]['context_items'] = json.loads(context_data['context_items'])
        self.context[stage]['context_tokens'] = json.loads(context_data['context_tokens'])
        self.context[stage]['context_words'] = json.loads(context_data['context_words'])
        self.context[stage]['interaction_history'] = json.loads(context_data['interaction_history'])
        self.context[stage]['user_profile'] = json.loads(context_data['user_profile'])
