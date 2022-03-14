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
import random

import numpy as np
import copy
from datetime import datetime
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, WARN, INFO
from collections import defaultdict

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq.token_generation_constraints import pack_constraints, unpack_constraints
from fairseq_cli.generate import get_symbols_to_strip_from_output

from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from page import *
 
from model_init import FavotModel,  add_local_args
from model import Favot

QUESTION_PATH = "/home/ubuntu/Platform/data/que/que.txt"

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


def test(logger, parser, args, cfg):

    fm = FavotModel(args, logger=logger)
    favot = Favot(args, fm, logger=logger, parser=parser)
    question_d = defaultdict(lambda: "")
    quesiton_l = []
    with open(QUESTION_PATH, "r") as f:
        for line in f:
            line = line.rstrip()
            quesiton_l.append(line)

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
                content = WEB_HTML.format(STYLE_SHEET, CSS, FONT_AWESOME, JS)
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
                    body["count"] = 5 if not "count" in body else body["count"]
                    body["question"] = "結婚生活は楽しいですか?" if not "question" in body else body["question"]

                    ret = favot.execute(body)
                    ret, ret_debug = ret
                    ret_ = [i[0] for i in ret.most_common(10)[:10]]
                    if ret is not None:
                        logger.info("sys_uttr: " + ret_[0])
                        print("\n".join(ret_debug))
                        print("sys: ", ret_)
                    print(ret, flush=True)
                    model_response = {"text": ret_}
                except Exception as e:
                    print("error", e, flush=True)
                    model_response = {"text": f"server error!!! 入力形式に誤りがあります。error Message: {e}"}
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                json_str = json.dumps(model_response)
                self.wfile.write(bytes(json_str, 'utf-8'))
            
            elif self.path == '/interact_one_res':
                content_length = int(self.headers['content-length'])
                try:
                    content = self.rfile.read(content_length).decode('utf-8')
                    print('body = {}'.format(content))
                    print(content_length)
                    # print('body = {}'.format(self.rfile.read(content_length).decode('utf-8')))
                    body = json.loads(content)

                    print('body = {}'.format(body))
                    body["count"] = 5 if not "count" in body else body["count"]
                    ID = "0"
                    if "ID" in body:
                        ID = body['ID']
                    if not question_d[ID]:
                        question_d[ID] = quesiton_l[random.randrange(len(quesiton_l))]
                    body["question"] = question_d[ID] if not "question" in body else body["question"]
                    
                    if "ID" in body:
                        print(f"対話,{body['ID']},{body['question']},{body['data']}", flush=True)

                    ret = favot.execute(body)
                    ret, ret_debug = ret
                   
                    if body["count"] == 1 and body["question"][:-1] not in ret:
                        ret = body["question"]

                    if ret is not None:
                        logger.info("sys_uttr: " + ret)
                        print("\n".join(ret_debug))
                        print("sys: " + ret)
                    print(ret, flush=True)
                    model_response = {"text": ret}
                except Exception as e:
                    print("error", e, flush=True)
                    model_response = {"text": f"server error!!! クラウドワークスにて連絡をお願いします。"}

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