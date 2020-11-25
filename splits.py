import os
import shutil
import csv
import pandas as pd
import argparse as ap

def generate_splits(data_path):
    data_path = data_path.rstrip(os.sep)
    train_path = os.path.join(data_path,"train")
    validation_path = os.path.join(data_path,"validation")
    test_path = os.path.join(data_path,"test")
    csv_path = os.path.join(data_path,os.path.basename(data_path)+".csv")

    print("Looking for " + csv_path)

    if not os.path.exists(csv_path):
        print("No csv found in " + data_path)
        return
    else:
        print("Found")

    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    if os.path.exists(validation_path):
        shutil.rmtree(validation_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)

    for path in [train_path, validation_path, test_path]:
        os.makedirs(path)

    df = pd.read_csv(csv_path, usecols=['split','midi_filename'])
    print("Generating splits...")
    for row_index,row in df.iterrows():
        if row['split']=='train':
            shutil.copyfile(os.path.join(data_path,row['midi_filename']), os.path.join(train_path, os.path.basename(row['midi_filename'])))
        elif row['split']=='validation':
            shutil.copyfile(os.path.join(data_path,row['midi_filename']), os.path.join(validation_path, os.path.basename(row['midi_filename'])))
        elif row['split']=='test':
            shutil.copyfile(os.path.join(data_path,row['midi_filename']), os.path.join(test_path, os.path.basename(row['midi_filename'])))
    print("Done")


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("data_path", help="path to data folder")
    args = parser.parse_args()
    generate_splits(args.data_path)
