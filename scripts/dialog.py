#coding: utf-8
"""
対応version
fairseq v0.10.2
python-telegram-bot v13.1
全体を依存少なくリライト

"""
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

SEPARATOR = "[SEP]"
SPK1 = "[SPK1]"
SPK2 = "[SPK2]"
hiragana = re.compile('[\u3041-\u309F，、．。？！\?\!]+')


def set_logger(name, rootname="log/main.log"):
    dt_now = datetime.now()
    dt = dt_now.strftime('%Y%m%d_%H%M%S')
    fname = rootname + "." + dt
    logger = getLogger(name)
    #handler1 = StreamHandler()
    #handler1.setFormatter(Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    handler2 = FileHandler(filename=fname)
    handler2.setLevel(DEBUG)  #handler2はLevel.WARN以上
    handler2.setFormatter(Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    #logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


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


class Favot(object):

    def encode_fn(self, x):
        if self.fm.tokenizer is not None:
            x = self.fm.tokenizer.encode(x)
        if self.fm.bpe is not None:
            x = self.fm.bpe.encode(x)
        return x

    def decode_fn(self, x):
        if self.fm.bpe is not None:
            x = self.fm.bpe.decode(x)
        if self.fm.tokenizer is not None:
            x = self.fm.tokenizer.decode(x)
        return x

    def __init__(self, args, favot_model, *, logger=None, parser=None):
        self.logger = logger
        self.parser = parser
        self.fm = favot_model
        self.args = args
        self.cfg = convert_namespace_to_omegaconf(args)
        self.contexts = []
        self.sent_contexts = []
        self.total_translate_time = 0
        self.debug = False
        self.delimiter = "．。 　?？!！♪☆★"
        self.sent_splitter = re.compile(".*?[{}]".format(self.delimiter), re.DOTALL)
        self.alphs = "abcdefghijklmnopqrstuvwyz"

        self.make_input_func = self.make_input

        utils.import_user_module(args)

        if args.buffer_size < 1:
            args.buffer_size = 1
        if args.max_tokens is None and args.batch_size is None:
            args.batch_size = 1

    def sent_split(self, line):
        _rets = self.sent_splitter.findall(line)
        rets = [r for r in _rets if r != ""]
        if "".join(rets) != line:
            c = re.sub(re.escape("".join(rets)), "", line)
            #c = c.strip(" \n\t")
            if c != "":
                rets.append(c)
        rets = [r.strip(" \n\t") for r in rets]
        return rets

    def common_word(self, word):
        word = word.strip("．。？?！!・")
        common = [
            "です",
            "ます",
            "ありがとう",
            "趣味",
            "(笑)",
        ]

        ## 本当はコーパス内の出現頻度で足きり
        if len(word) <= 1:
            return True
        if len(word) <= 2:
            hira = hiragana.findall(word)
            if len(hira) == 0:
                pass
            elif len("".join(hira)) >= 1:
                return True
            if word in ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]:
                return True
        if len(word) <= 3:
            if word[-1] == "い" or word[-1] == "る":
                return True
        for c in common:
            if c in word:
                return True
        if hiragana.fullmatch(word) is not None:
            return True

        return False

    def set_generator_parameters(self, args):
        for k, v in args.items():
            #_args = self.parser.parse_args(["--"+k, v])
            cur_v = self.args.__dict__[k]
            if v == "None":
                self.args.__setattr__(k, None)
            elif type(cur_v) == int:
                self.args.__setattr__(k, int(v))
            elif type(cur_v) == float:
                self.args.__setattr__(k, float(v))
            elif type(cur_v) == bool:
                if v == "False" or v == "false":
                    self.args.__setattr__(k, False)
                else:
                    self.args.__setattr__(k, True)
            elif type(cur_v) == str:
                self.args.__setattr__(k, str(v))
            else:
                raise TypeError("Unknown type of generator parameter")
            #self.args.__setattr__(k, _args.__dict__[k])
            print(self.args)
        self.fm.generator = self.fm.task.build_generator(self.fm.models, self.args)
        _args = copy.deepcopy(self.args)
        _args.__setattr__("score_reference", True)
        _args.__setattr__("beam", 1)
        _args.__setattr__("nbest", 1)

        self.fm.scorer = self.fm.task.build_generator(self.fm.models, _args)
        self.logger.info("update generator parameter:" + str(args))
        return

    def make_single_sample(self, inputs, args, task, max_positions):
        ret = []

        for batch in make_batches(inputs, args, task, max_positions, self.encode_fn):
            bsz = batch.src_tokens.size(0)
            tokens = batch.src_tokens
            lengths = batch.src_lengths
            constraints = batch.constraints
            if self.fm.use_cuda:
                tokens = tokens.cuda()
                lengths = lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                'net_input': {
                    'src_tokens': tokens,
                    'src_lengths': lengths,
                    'prev_output_tokens': tokens,
                },
            }
            ret.append(sample)
        return ret

    def execute(self, uttr, mode="normal"):
        ret = self._execute(uttr, mode=mode)
        if ret is not None:
            ret_scores, ret_debug = ret
        else:
            return
        if len(ret_scores) == 0:
            return "", ret_debug
        ret_utt, ret_score = ret_scores.most_common(1)[0]
        print(ret_score, ret_utt)

        self.logger.info(str(ret_scores.most_common(5)))
        return ret_utt, ret_debug

    def _execute(self, uttr, **kwargs):
        mode = "normal"
        if "mode" in kwargs:
            mode = kwargs["mode"]

        ret_debug = []
        start_time = time.time()
        start_id = 0

        inputs = [
            self.make_input_func(SPK2, uttr, mode=mode),
        ]

        self.logger.info("input_seq: " + str(inputs))
        if self.debug:
            ret_debug.append("input_seq: " + str(inputs))
        results = []

        args = self.fm.cfg
        task = self.fm.task
        max_positions = self.fm.max_positions
        use_cuda = self.fm.use_cuda

        for i, batch in enumerate(make_batches(inputs, args, task, max_positions, self.encode_fn)):
            bsz = batch.src_tokens.size(0)
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            constraints = batch.constraints
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
                if constraints is not None:
                    constraints = constraints.cuda()

            sample = {
                'net_input': {
                    'src_tokens': src_tokens,
                    'src_lengths': src_lengths,
                    'prev_output_tokens': src_tokens
                },
                #"target": zero_samples[i]["net_input"]["src_tokens"],
            }
            translate_start_time = time.time()
            translations = task.inference_step(self.fm.generator, self.fm.models, sample, constraints=constraints)
            translate_time = time.time() - translate_start_time
            self.fm.total_translate_time += translate_time
            list_constraints = [[] for _ in range(bsz)]

            if args.generation.constraints:
                list_constraints = [unpack_constraints(c) for c in constraints]
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], self.fm.tgt_dict.pad())
                constraints = list_constraints[i]
                results.append((start_id + id, src_tokens_i, hypos, {
                    "constraints": constraints,
                    "time": translate_time / len(translations)
                }))

        ret_cands = []
        ret_scores = collections.Counter()
        # sort output to match input order
        for id_, src_tokens, hypos, info in sorted(results, key=lambda x: x[0]):
            if self.fm.src_dict is not None:
                src_str = self.fm.src_dict.string(src_tokens, args.common_eval.post_process)
                #src_str = self.fm.src_dict.string(src_tokens, args.post_process)
                print("W-{}\t{:.3f}\tseconds".format(id_, info["time"]))
                for constraint in info["constraints"]:
                    print("C-{}\t{}".format(id_, self.fm.tgt_dict.string(constraint, args.common_eval.post_process)))
                    if self.debug:
                        ret_debug.append("C-{}\t{}".format(
                            id_, self.fm.tgt_dict.string(constraint, args.common_eval.post_process)))
            # Process top predictions

            _cand_counter = collections.Counter()
            for i, hypo in enumerate(hypos[:min(len(hypos), min(args.generation.nbest, self.args.show_nbest))]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=self.fm.align_dict,
                    tgt_dict=self.fm.tgt_dict,
                    remove_bpe=args.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.fm.generator),
                )
                detok_hypo_str = self.decode_fn(hypo_str)
                _cand = detok_hypo_str

                score = hypo['score'] / math.log(2)  # convert to base 2

                # remove duplicate candidates
                dup_flag, nodup_cand = self.contain_duplicate(detok_hypo_str, mode=mode, id=id_)
                #ret_scores[detok_hypo_str] = score - 10000
                if dup_flag and self.args.suppress_duplicate:
                    self.logger.info("duplicated pattern: {}".format(detok_hypo_str))
                    if nodup_cand != "":
                        self.logger.info("no dup cand: {}".format(nodup_cand))
                        score -= 100
                    else:
                        score = score - 100000
                # original hypothesis (after tokenization and BPE)
                self.logger.info("system_utt_cands: " + 'H-{}\t{}\t{}'.format(id_, score, hypo_str))
                self.logger.info("system_utt_cands: " + 'D-{}\t{}\t{}'.format(id_, score, detok_hypo_str))
                if self.debug:
                    #ll='H-{}\t{}\t{}'.format(id_, score, hypo_str)+"\n"+'D-{}\t{}\t{}'.format(id_, score, detok_hypo_str)
                    #ret_debug.append(ll)
                    ret_debug.append("system_utt_cands: " + 'D-{}\t{}\t{}'.format(id_, score, detok_hypo_str))

                _scores = hypo['positional_scores'].div_(math.log(2)).tolist()
                _contexts = self.contexts

                if "<ex>" in detok_hypo_str:
                    detok_hypo_str = detok_hypo_str.replace("<ex>", "").replace("</ex>", "")
                    if "<" in detok_hypo_str or ">" in detok_hypo_str:
                        score -= 1000
                detok_hypo_str = detok_hypo_str.replace("(笑)", " ").replace("(笑）", " ").replace("（笑)", " ")
                if "unk" in detok_hypo_str and len(_contexts) > 0:
                    c1 = re.findall("(..<unk>)", detok_hypo_str)
                    c2 = re.findall("(.<unk>.)", detok_hypo_str)
                    c3 = re.findall("(..<unk>)", detok_hypo_str)
                    self.logger.info("{}/{}/{}".format(str(c1), str(c2), str(c3)))
                    try:
                        if len(c1) > 0:
                            c1 = c1[0]
                            cc = re.findall("{}(.)".format(c1[0:2]), _contexts[-1]["utt"], re.DOTALL)
                            if len(cc) > 0:
                                detok_hypo_str = detok_hypo_str.replace(c1[0:2] + "<unk>", c1[0:2] + cc[0])
                        elif len(c2) > 0:
                            c2 = c2[0]
                            cc = re.findall("{}(.){}".format(c2[0], c2[1]), _contexts[-1]["utt"], re.DOTALL)
                            if len(cc) > 0:
                                detok_hypo_str = detok_hypo_str.replace(c2[0] + "<unk>" + c2[1], c2[0] + cc[0] + c2[1])
                        elif len(c3) > 0:
                            c3 = c3[0]
                            cc = re.findall("(.){}".format(c3[0:2]), _contexts[-1]["utt"], re.DOTALL)
                            if len(cc) > 0:
                                detok_hypo_str = detok_hypo_str.replace("<unk>" + c3[0:2], cc[0] + c3[0:2])
                        else:
                            score -= 1000
                    except:
                        score -= 1000
                if "呼べば" in detok_hypo_str or "呼ん" in detok_hypo_str or "呼び" in detok_hypo_str:
                    score -= 2
                if mode != "prefinish" and mode != "finish":
                    if "時間で" in detok_hypo_str:
                        score -= 2
                        if "そろそろ" in detok_hypo_str:
                            score -= 1000000
                # if self.args.rep_pen != 0:
                #     repeat_num = self.num_repeat_topic_word(detok_hypo_str, mode=mode, contexts=_contexts)
                #     score -= repeat_num * self.args.rep_pen
                # #suspect, contained = self.cooccur_check(detok_hypo_str)
                # if self.args.sus_pen != 0 or self.args.check_reward != 0:
                #     suspect_num, checked_num = self.cooccur_check(detok_hypo_str, mode=mode, contexts=_contexts)
                #     score += min(checked_num, 2) * self.args.check_reward  # 0.5?
                #     score -= suspect_num * self.args.sus_pen
                # #suspect_num = len(suspect)
                # #contained_num = len(contained)
                # #score -= sum([detok.hypo_str.count(c) - 1 for c in contained])
                # score -= detok_hypo_str.count("、") * self.args.toks_pen
                nodup_cand = nodup_cand.replace("(笑)", " ").replace("(笑）", " ").replace("（笑)", " ")

                # if self.args.nodup:
                #     ret_scores[nodup_cand] = score
                # else:
                #     #_cand_counter[detok_hypo_str] = score
                ret_scores[detok_hypo_str] = score

                self.logger.info("system_utt_cands: " + 'P-{}\t{}'.format(
                    id_,
                    ' '.join(
                        map(
                            lambda x: '{:.4f}'.format(x),
                            # convert from base e to base 2
                            hypo['positional_scores'].div_(math.log(2)).tolist(),
                        ))))

                if args.generation.print_alignment:
                    alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                    print('A-{}\t{}'.format(id_, alignment_str))

        return ret_scores, ret_debug

    def contain_duplicate(self, hypo, mode="normal", id=-1):
        #sents = self.sent_splitter.findall(hypo)
        sents = self.sent_split(hypo)
        nodup_cand = []
        ff = False
        sent_contexts = self.sent_contexts
        contexts = self.contexts
        for orgs in sents:
            f = False
            s = orgs.rstrip("!?！？。．　・")
            spk2_skip = 0
            for i, cdic in enumerate(sent_contexts[::-1]):
                if cdic["spk"] == SPK2 and spk2_skip < 2:
                    continue
                elif cdic["spk"] == SPK1:
                    spk2_skip += 1

                c = cdic["utt"].rstrip("!?！？。．　・")
                ## remove too short sentences with no hiragana
                hiras = hiragana.findall(s)
                hira = "".join(hiras)
                if len(hira) >= len(c) - 1 and (len(c) < 7 or len(s) < 7):
                    continue
                if "そう" in c and len(c) < 10:
                    continue
                e = difflib.SequenceMatcher(None, s, c).ratio()
                if e > 0.5:
                    self.logger.info("sim: {}, cand: {}, contexts: {}".format(e, s, c))
                if e > 0.65:
                    f = True
                    ff = True
                    break
            if not f:
                nodup_cand.append(orgs)

        ## 文全体チェック: nodup_candでかけるように変更
        f = False
        for cdic in contexts:
            if cdic["spk"] == SPK2:
                continue
            c = cdic["utt"]
            #e = difflib.SequenceMatcher(None, hypo, c).ratio()
            e = difflib.SequenceMatcher(None, "".join(nodup_cand), c).ratio()
            if e > 0.5:
                self.logger.info("all sim: {}, cand: {}, contexts: {}".format(e, hypo, c))
            if e > 0.5:
                f = True
                ff = True
                break

        ## check duplicate tokens within the sentence itself
        _contexts = []
        #for i, s in enumerate(sents):
        for i, s in enumerate(nodup_cand):
            _contexts.append({"spk": SPK1, "utt": s, "id": i})

        #for i, s in enumerate(sents):
        new_nodup_cand = []
        for i, s in enumerate(nodup_cand):
            f = False
            for j, cdic in enumerate(_contexts):
                c = cdic["utt"]
                ## skip too short sentences
                s = s.strip(" ")
                c = c.strip(" ")

                if len(c) < 2:
                    continue
                e = difflib.SequenceMatcher(None, s, c).ratio()
                if i == j:
                    continue
                if e > 0.5:
                    self.logger.info("self: sim: {}, cand: {}, contexts: {}".format(e, s, c))
                if e > 0.65:
                    f = True
                    ff = True
                    break
            if not f:
                new_nodup_cand.append(s)
        ret_flag = ff
        return ret_flag, "".join(new_nodup_cand)

    def add_contexts(self, spk, utt, mode="normal"):
        self._add_contexts(spk, utt)
        return

    def _add_contexts(self, spk, utt):
        self.contexts.append({"spk": spk, "utt": utt})
        for s in self.sent_split(utt):
            self.sent_contexts.append({"spk": spk, "utt": s})
        return

    def make_input(self, newspk, newutt, mode="normal", max_contexts=-1, id=None, idprefix="a"):
        if max_contexts == -1:
            max_contexts = self.args.max_contexts
        line = ""

        contexts = newutt
        print(contexts, flush=True)
        SP = {"U": SPK2, "S": SPK1, "s":SPK1, "u":SPK2}
        line = [SP[_context["Talker"][0]] + _context["Uttr"] for _context in contexts["data"]]
        print(line, flush=True)
        return SEPARATOR.join(line)[-512:]

    def reset(self):
        self.contexts = []
        self.sent_contexts = []
        return


STYLE_SHEET = "https://cdnjs.cloudflare.com/ajax/libs/bulma/0.7.4/css/bulma.css"
FONT_AWESOME = "https://use.fontawesome.com/releases/v5.3.1/js/all.js"
WEB_HTML = """
<html>
    <link rel="stylesheet" href={} />
    <script defer src={}></script>
    <head><title> Interactive Run </title></head>
    <body>
        <div class="columns" style="height: 100%">
            <div class="column is-three-fifths is-offset-one-fifth">
              <section class="hero is-info is-large has-background-light has-text-grey-dark" style="height: 100%">
                <div id="parent" class="hero-body" style="overflow: auto; height: calc(100% - 76px); padding-top: 1em; padding-bottom: 0;">
                    <article class="media">
                      <div class="media-content">
                        <div class="content">
                          <p>
                            <strong>Instructions</strong>
                            <br>
                            Enter a message, and the model will respond interactively. <br>
                            日本語で入力してもらっても大丈夫です．<br>
                            会話をやり直す場合はブラウザを更新してください．<br>
                            会話を入力してから，若干ラグがありますが，基本的には大丈夫です．ただ，5秒以上たっても返信がない場合は horiso0921@gmail.com まで連絡ください<br>
                        </p>
                        </div>
                      </div>
                    </article>
                </div>
                <div class="hero-foot column is-three-fifths is-offset-one-fifth" style="height: 76px">
                  <form id = "interact">
                      <div class="field is-grouped">
                        <p class="control is-expanded">
                          <input class="input" type="text" id="userIn" placeholder="Type in a message">
                        </p>
                        <p class="control">
                          <button id="respond" type="submit" class="button has-text-white-ter has-background-grey-dark">
                            Submit
                          </button>
                        </p>
                      </div>
                  </form>
                </div>
              </section>
            </div>
        </div>
        <script>
            function createChatRow(agent, text) {{
                var article = document.createElement("article");
                article.className = "media"
                var figure = document.createElement("figure");
                figure.className = "media-left";
                var span = document.createElement("span");
                span.className = "icon is-large";
                var icon = document.createElement("i");
                icon.className = "fas fas fa-2x" + (agent === "You" ? " fa-user " : agent === "Model" ? " fa-robot" : "");
                var media = document.createElement("div");
                media.className = "media-content";
                var content = document.createElement("div");
                content.className = "content";

                var para2 = document.createElement("p");
                var paraText = document.createTextNode(text);
                para2.className = "utt";

                var para1 = document.createElement("p");
                var strong = document.createElement("strong");
                strong.innerHTML = agent;
                var br = document.createElement("br");

                para1.appendChild(strong);
                para2.appendChild(paraText);
                content.appendChild(para1);
                content.appendChild(para2);
                media.appendChild(content);

                span.appendChild(icon);
                figure.appendChild(span);

                if (agent !== "Instructions") {{
                    article.appendChild(figure);
                }};
                article.appendChild(media);
                return article;
            }}
            var parDiv = document.getElementById("parent");
            parDiv.append(createChatRow("Model", "こんにちは。よろしくお願いします。"));
            parDiv.scrollTo(0, parDiv.scrollHeight);
            var context = [
                {{"Talker": "S",
                 "Uttr":   "こんにちは。よろしくお願いします。"
                }}];
            document.getElementById("interact").addEventListener("submit", function(event){{
                event.preventDefault()

                ntex = document.getElementById("userIn").value;
                var ncontext = {{"Talker": "U",
                 "Uttr":   ntex
                }};
                context.push(ncontext);

                document.getElementById('userIn').value = "";
                var parDiv = document.getElementById("parent");
                parDiv.append(createChatRow("You", ntex));
                parDiv.scrollTo(0, parDiv.scrollHeight);
                var send_info = {{"data": context}};
                fetch('/interact', {{
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    method: 'POST',
                    body: JSON.stringify(send_info)
                }}).then(response=>response.json()).then(data=>{{
                    // Change info for Model response
                    parDiv.append(createChatRow("Model", data.text));
                    parDiv.scrollTo(0, parDiv.scrollHeight);
                    context.push({{"Talker": "S", "Uttr": data.text}});
                }})
            }});
            document.getElementById("interact").addEventListener("reset", function(event){{
                event.preventDefault()
                var text = document.getElementById("userIn").value;
                document.getElementById('userIn').value = "";
                fetch('/reset', {{
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    method: 'POST',
                }}).then(response=>response.json()).then(data=>{{
                    var parDiv = document.getElementById("parent");
                    parDiv.innerHTML = '';
                    parDiv.append(createChatRow("Instructions", "Enter a message, and the model will respond interactively."));
                    parDiv.scrollTo(0, parDiv.scrollHeight);
                }})
            }});
        </script>
    </body>
</html>
"""  # noqa: E501

        

def add_local_args(parser):
    parser.add_argument('--max-contexts', type=int, default=4, help='max length of used contexts')
    parser.add_argument('--suppress-duplicate', action="store_true", default=False, help='suppress duplicate sentences')
    parser.add_argument('--show-nbest', default=3, type=int, help='# visible candidates')
    parser.add_argument('--starting-phrase', default="こんにちは。よろしくお願いします。", type=str, help='starting phrase')
    return parser


def test(logger, parser, args, cfg):

    fm = FavotModel(args, logger=logger)
    favot = Favot(args, fm, logger=logger, parser=parser)

    class MyHTTPRequestHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            paths = {
            '/': {'status': 200},
            '/favicon.ico': {'status': 202},  # Need for chrome
            }
            if not self.path in paths:
                response = 500
                self.send_response(response)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                content = WEB_HTML.format(STYLE_SHEET, FONT_AWESOME)
                self.wfile.write(bytes(content, 'UTF-8'))
            else:
                response = paths[self.path]['status']
                print('path = {}'.format(self.path))

                parsed_path = urlparse(self.path)
                print('parsed: path = {}, query = {}'.format(parsed_path.path, parse_qs(parsed_path.query)))

                print('headers\r\n-----\r\n{}-----'.format(self.headers))

                self.send_response(response)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                content = WEB_HTML.format(STYLE_SHEET, FONT_AWESOME)
                self.wfile.write(bytes(content, 'UTF-8'))

        

        def do_POST(self):
            """
            Handle POST request, especially replying to a chat message.
            """
            print('path = {}'.format(self.path))
            parsed_path = urlparse(self.path)
            print('parsed: path = {}, query = {}'.format(parsed_path.path, parse_qs(parsed_path.query)))

            print('headers\r\n-----\r\n{}-----'.format(self.headers))

            if self.path == '/interact':
                content_length = int(self.headers['content-length'])
                try:
                    content = self.rfile.read(content_length).decode('utf-8')
                    print('body = {}'.format(content))
                    print(content_length)
                    # print('body = {}'.format(self.rfile.read(content_length).decode('utf-8')))
                    body = json.loads(content)

                    print('body = {}'.format(body))

                    ret = favot.execute(body)
                    ret, ret_debug = ret
                    if ret is not None:
                        logger.info("sys_uttr: " + ret)
                        print("\n".join(ret_debug))
                        print("sys: " + ret)
                    model_response = {"text": ret}
                except Exception as e:
                    print("error", e, flush=True)
                    model_response = {"text": f"server error!!! 入力形式に誤りがあります。error Message: {e}"}

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                json_str = json.dumps(model_response)
                self.wfile.write(bytes(json_str, 'utf-8'))

    print("Start", flush=True)
    address = ('localhost', 8080)

    MyHTTPRequestHandler.protocol_version = 'HTTP/1.0'
    with HTTPServer(address, MyHTTPRequestHandler) as server:
        server.serve_forever()


def main():
    logger = set_logger("dialog", "log/dialog.log")
    parser = options.get_interactive_generation_parser()
    add_local_args(parser)
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    test(logger, parser, args, cfg)


if __name__ == "__main__":
    main()