import os
import argparse


def prepare_training_folder(path, pos):
    for filename in os.listdir(str(path)):
        file_path = str(path) + "/" + filename
        if pos == path:
            with open('info.dat', 'a') as f:
                f.write(file_path + " 1 0 0 28 28\n")
        else:
            with open('bg.txt', 'a') as f:
                f.write(file_path + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels', default=0)
    args = parser.parse_args()
    positive = int(args.labels)
    print(positive)
    for f in ['info.dat', 'bg.txt']:
        pfile = open(f, "w+")
        pfile.seek(0)
        pfile.truncate()
    for i in range(0, 10):
        prepare_training_folder(i, positive)
