#!/usr/bin/env python

import argparse
import logging

from libs.transformer_class import Seq2SeqModel
from kserve import ModelServer, model_server

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(parents=[model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    try:
        model = Seq2SeqModel(args.model_name)
        ModelServer(args.http_port,
                    args.max_threads,
                    args.enable_docs_url).start([model])
    except Exception:
        logger.exception("Failed to start model server")
        raise
