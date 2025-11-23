import argparse

from RunModel import run_model
from model import MFSC_DTI

parser = argparse.ArgumentParser(
    prog='MFSC_DTI',
    description='MFSC_DTI is model in paper: \"Multi-Level Feature Extraction and Multi-Channel Shared Cross-Attention Framework for DTI prediction\"',
    epilog='Model config set by config.py')

parser.add_argument('dataSetName', choices=[
                    "DrugBank","Enzyme", "GPCRs"],
                    help='Enter which dataset to use for the experiment')
parser.add_argument('-m', '--model', choices=['MFSC_DTI'],
                    default='MFSC_DTI', help='Which model to use, \"MFSC_DTI\" is used by default')
parser.add_argument('-s', '--seed', type=int, default=114514,
                    help='Set the random seed, the default is 114514')
parser.add_argument('-f', '--fold', type=int, default=5,
                    help='Set the K-Fold number, the default is 5')

args = parser.parse_args()

# 修改调用部分
if args.model == 'MFSC_DTI':
    run_model(SEED=args.seed,
              DATASET=args.dataSetName,
              MODEL=MFSC_DTI,
              K_Fold=args.fold)

