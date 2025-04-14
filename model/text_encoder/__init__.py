import os, sys

CUR_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(CUR_DIR, os.pardir))

sys.path.extend([MODEL_DIR, CUR_DIR])