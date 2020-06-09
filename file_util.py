import json
import pickle
import os
import os.path as op
import errno
import shutil
import gzip
import collections
import numpy as np
import csv


def load_data(input_file_name):
    ext = op.splitext(input_file_name)[1]

    assert ext in (".json", ".pkl", ".npy", ".txt", ".csv")

    data = None

    if op.isfile(input_file_name):
        if ext == ".json":
            with open(input_file_name, 'r') as in_file:
                data = json.load(in_file, object_pairs_hook=collections.OrderedDict)
        elif ext == ".pkl":
            with open(input_file_name, "rb") as in_file:
                data = pickle.load(in_file)
        elif ext == ".npy":
            data = np.load(input_file_name)
        elif ext == ".csv":
            data = []
            with open(input_file_name, "r") as in_file:
                csv_file = csv.reader(in_file)
                for data_item in csv_file:
                    data.append(data_item)
        else:
            data = []
            with open(input_file_name, "r") as in_file:
                for line in in_file:
                    data.append(line.strip())

    return data


def save_data(data, output_file_name):
    ext = op.splitext(output_file_name)[1]

    assert ext in (".json", ".pkl", ".npy", ".csv")

    if ext == ".json":
        with open(output_file_name, "w") as out_file:
            json.dump(data, out_file, indent=2)
    elif ext == ".pkl":
        with open(output_file_name, "wb") as out_file:
            pickle.dump(data, out_file, -1)
    elif ext == ".npy":
        np.save(output_file_name, data)
    elif ext == ".csv":
        with open(output_file_name, "w") as out_file:
            csv_file = csv.writer(out_file)
            for data_item in data:
                csv_file.writerow(data_item)
    else:
        with open(output_file_name, "w") as out_file:
            for item in data:
                out_file.write(item)
                out_file.write("\n")


def make_sure_path_exists(path):
    if len(path) > 0:  # skip empty dir
        try:
            os.makedirs(path)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise


def copy_files(source_dir, target_dir):
    make_sure_path_exists(target_dir)
    for file_name in os.listdir(source_dir):
        full_file_name = op.join(source_dir, file_name)
        if op.isfile(full_file_name):
            shutil.copy(full_file_name, target_dir)


def gz_uncompress(source_file_name, target_file_name):
    assert source_file_name.endswith(".gz")
    with gzip.open(source_file_name, 'rb') as f_in, open(target_file_name, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)


def gz_compress(source_file_name, target_file_name):
    assert target_file_name.endswith(".gz")
    with open(source_file_name, 'rb') as f_in, gzip.open(target_file_name, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

