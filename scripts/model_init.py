"何が必要かわからなかったのでとりあえず全部インポートしている"

#from fairseq_cli import interactive as intr
from fairseq_cli.interactive import make_batches
import sys
import ast
import torch
import time
import math
import re
import difflib
import collections
import json

import numpy as np
import copy
from datetime import datetime
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, WARN, INFO

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output

from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from page import *

def add_local_args(parser):
    parser.add_argument('--max-contexts', type=int, default=4,
                        help='max length of used contexts')
    parser.add_argument('--suppress-duplicate', action="store_true",
                        default=False, help='suppress duplicate sentences')
    parser.add_argument('--show-nbest', default=3,
                        type=int, help='# visible candidates')
    parser.add_argument(
        '--starting-phrase', default="こんにちは。よろしくお願いします。", type=str, help='starting phrase')
    parser.add_argument('--quemode', default="off",
                        type=str, help='question mode')
    parser.add_argument('--que', default="あなたの宗教は何ですか？",
                        type=str, help='question mode')
    parser.add_argument('--turn', default=9, type=int, help='question mode')
    parser.add_argument('--savepath', default="log", type=str, help='question mode')
    parser.add_argument('--basepath', default="/data/group1/z44384r/finetuning-nttdialoguemodel/model/base/empdial50k-flat_1.6B_19jce27w_3.86.pt", type=str, help='base model path')
    parser.add_argument('--saya', default=False, type=bool, help='Is saya model')
    parser.add_argument('--train-turn', default=5, type=int, help='学習時のターン数')
    return parser

class FavotModel(object):

    def __init__(self, args, *, logger=None):
        self.logger = logger
        self.args = args
        self.cfg = None
        #if not legacymode:
        self.cfg = convert_namespace_to_omegaconf(args)
        cfg = self.cfg
        #self.cfg.generation.constraints = args.constraints

        if hasattr(self.args, "remove_bpe"):
            self.args.post_process = self.args.remove_bpe
        else:
            self.args.remove_bpe = self.args.post_process
        self.contexts = []
        start_time = time.time()
        self.total_translate_time = 0
        utils.import_user_module(args)

        if args.buffer_size < 1:
            args.buffer_size = 1
        #if args.max_tokens is None and args.max_sentences is None:
        if args.max_tokens is None and args.batch_size is None:
            args.max_sentences = 1
            args.batch_size = 1

        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'
        #assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        #    '--max-sentences/--batch-size cannot be larger than --buffer-size'
        print(args.batch_size, args.buffer_size, args.batch_size <= args.buffer_size)
        assert not args.batch_size or args.batch_size <= args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        self.logger.info(cfg)

        # Fix seed for stochastic decoding
        if args.seed is not None and not args.no_seed_provided:
            np.random.seed(args.seed)
            utils.set_torch_seed(args.seed)

        self.use_cuda = torch.cuda.is_available() and not args.cpu

        # Setup task, e.g., translation
        #if legacymode:
        self.task = tasks.setup_task(args)
        #else:
        #    self.task = tasks.setup_task(self.cfg)

        # Load ensemble
        self.logger.info('loading model(s) from {}'.format(args.path))

        #return
        overrides = ast.literal_eval(cfg.common_eval.model_overrides)
        logger.info("loading model(s) from {}".format(cfg.common_eval.path))
        self.models, self._model_args = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=self.task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )
        # self.models, self._model_args = checkpoint_utils.load_model_ensemble(
        #     args.path.split(os.pathsep),
        #     arg_overrides=eval(args.model_overrides),
        #     task=self.task,
        #     suffix=getattr(args, "checkpoint_suffix", ""),
        # )

        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Optimize ensemble for generation
        for model in self.models:
            #if legacymode:
            #    model.prepare_for_inference_(args)
            #else:
            model.prepare_for_inference_(self.cfg)
            if args.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()

        # Initialize generator
        self.generator = self.task.build_generator(self.models, args)
        _args = copy.deepcopy(args)
        _args.__setattr__("score_reference", True)
        self.scorer = self.task.build_generator(self.models, _args)
        # Handle tokenization and BPE
        self.tokenizer = encoders.build_tokenizer(args)
        self.bpe = encoders.build_bpe(args)

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(args.replace_unk)

        self.max_positions = utils.resolve_max_positions(self.task.max_positions(),
                                                         *[model.max_positions() for model in self.models])

        if self.cfg.generation.constraints:
            logger.warning("NOTE: Constrained decoding currently assumes a shared subword vocabulary.")

        if self.cfg.interactive.buffer_size > 1:
            logger.info("Sentence buffer size: %s", self.cfg.interactive.buffer_size)

        # if args.constraints:
        #     self.logger.warning("NOTE: Constrained decoding currently assumes a shared subword vocabulary.")

        # if args.buffer_size > 1:
        #     self.logger.info('Sentence buffer size: %s', args.buffer_size)
        #logger.info('NOTE: hypothesis and token scores are output in base 2')
        #logger.info('Type the input sentence and press return:')
        self.logger.info("loading done")
        #print("loading done")
