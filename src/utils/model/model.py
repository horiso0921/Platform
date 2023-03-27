import ast
import torch

import numpy as np
import copy

from fairseq import checkpoint_utils, tasks, utils
from fairseq.data import encoders

from fairseq.dataclass.utils import convert_namespace_to_omegaconf

def add_local_args(parser):
    parser.add_argument('--max-contexts', type=int, default=4,
                        help='max length of used contexts')
    parser.add_argument('--suppress-duplicate', action="store_true",
                        default=False, help='suppress duplicate sentences')

    return parser

class Model(object):

    def __init__(self, args, *, logger=None):
        """_summary_

        Args:
            args (_type_): _description_
            logger (_type_, optional): _description_. Defaults to None.
        """
        
        self.logger = logger
        self.args = args
        self.cfg = convert_namespace_to_omegaconf(args)
        cfg = self.cfg
        
        if hasattr(self.args, "remove_bpe"):
            self.args.post_process = self.args.remove_bpe
        else:
            self.args.remove_bpe = self.args.post_process
        
        self.contexts = []
        # utils.import_user_module(args)

        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.batch_size is None:
            args.max_sentences = 1
            args.batch_size = 1

        assert not args.sampling or args.nbest == args.beam, \
            '--sampling requires --nbest to be equal to --beam'

        assert not args.batch_size or args.batch_size <= args.buffer_size, \
            '--max-sentences/--batch-size cannot be larger than --buffer-size'

        self.logger.info(cfg)

        # Fix seed for stochastic decoding
        if args.seed is not None and not args.no_seed_provided:
            np.random.seed(args.seed)
            utils.set_torch_seed(args.seed)

        self.use_cuda = torch.cuda.is_available() and not args.cpu

        # Setup task, e.g., translation
        self.task = tasks.setup_task(args)
        
        # Load ensemble
        self.logger.info('loading model(s) from {}'.format(args.path))

        # 文字列を辞書にしているらしい
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

        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        # Optimize ensemble for generation
        for model in self.models:
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

        self.logger.info("loading done")
        