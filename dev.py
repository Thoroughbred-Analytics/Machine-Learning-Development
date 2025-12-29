from model.model import *
from model.util import *
from model.train import *
from model.predict import *
from model import model
import torch
import pandas as pd
import argparse
import sys

def create_parser():
    parser = argparse.ArgumentParser(description="Horse Predictor Rating")

    subparsers = parser.add_subparsers(
        dest='command',
        title='commands',
        description='Available commands',
        help='Choose a command to run',
        required=True
    )

    # Training argument
    train_parser = subparsers.add_parser('train',
                                         help='Train a new model',
                                         description='Train a new model with a specific dataset.')
    train_parser.add_argument("data",
                              help="the path to the processed csv file to be trained")
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=64,
                             help='Training batch size')
    train_parser.add_argument('--lr', type=float, default=0.1e-4,
                             help='Learning rate')
    train_parser.add_argument('--output', type=str, default='model/trainedModels/model',
                             help='Output model path')
    
    predict_parser = subparsers.add_parser('predict',
                                           help='Make a random prediction',
                                           description='With a trained model, make a random prediction with samples within'
                                           'the dataset to visualize model predictions vs actual ratings.')
    predict_parser.add_argument('--model', type=str, default="model/trainedModels/base.pth",
                                help='Path to saved model')
    predict_parser.add_argument('--num-samples', type=int, default=20,
                                help='Number of random samples to predict and plot')
    predict_parser. add_argument('--dataset', type=str, default="data/baseData.csv",
                                 help='path to the dataset we want to use to predict')
    predict_parser.add_argument('--graph', type=bool, default=False,
                               help='show a plot of the predicted values vs the actual values.')
    
    
    dataset_parser = subparsers.add_parser('create-dataset',
                                           help='create a new dataset from csv file',
                                           description='create a clean dataset for model training from an ' \
                                           'input .csv file.')
    dataset_parser.add_argument("input_path",
                                help="path to the dataset that will be processed")
    dataset_parser.add_argument("output_dataset",
                                help="name of the output dataset. ")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "create-dataset":

        try:
            df, _ = preprocess_csv(args.input_path)
            df.to_csv(args.output_dataset, index=False)

        except FileNotFoundError:
            print(f"Error: File '{args.input_path}' not found.")

    elif args.command == "train":
        try:
            trainModel(args.data, args.epochs, args.output, args.lr, args.batch_size)

        except FileNotFoundError:
            print("Invalid file sequence. refer to -h for more information. ")
        

    elif args.command == "predict":
        try:
            predict_samples(args.model, args.dataset, args.graph, args.num_samples)
        except FileNotFoundError:
            print("File not found")
            
if __name__ == "__main__":
    main()