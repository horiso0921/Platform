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

from fairseq import options
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse
from model_init import FavotModel,  add_local_args
from model_interact import InteractFavot as Favot
from page import *


def set_logger(name, rootname="../log/main.log"):
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


def test(logger, parser, args):
    fm = FavotModel(args, logger=logger)
    favot = Favot(args, fm, logger=logger, parser=parser)

    while True:
        line = input("content (Type end to end) >> ").rstrip("\n")

        if line.startswith("end"):
            break

        ret = favot.execute(line)
        if ret is None:
            continue
        ret, ret_debug = ret
        if ret is not None:
            logger.info("sys_uttr: " + ret)
            print("\n".join(ret_debug))
            print("sys: " + ret)


def main():
    logger = set_logger("dialog", "../log/dialog.log")
    parser = options.get_interactive_generation_parser()
    add_local_args(parser)
    args = options.parse_args_and_arch(parser)
    test(logger, parser, args)

if __name__ == "__main__":
    main()