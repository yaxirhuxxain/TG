import argparse
import os
import random

from tqdm import tqdm


def split(projects_info, type="file", ratio="50-20-30"):
    train_ratio, dev_ratio, test_ratio = ratio.split("-")
    train_ratio, dev_ratio, test_ratio = int(
        train_ratio) / 100, int(dev_ratio) / 100, int(test_ratio) / 100

    # raise an error if ratio not equal to 1
    assert train_ratio + dev_ratio + test_ratio == 1

    print(
        f"Creating Splits with {type} ratio of "
        f"train: {train_ratio * 100}%, dev: {dev_ratio * 100}%, and test: {test_ratio * 100}% "
    )

    output = {
        "train": {"projects": [], f"count_{type}": 0},
        "dev": {"projects": [], f"count_{type}": 0},
        "test": {"projects": [], f"count_{type}": 0}
    }

    files_count = {}
    for project, info in projects_info.items():
        files_count[project] = info['files_count']

    total = sum(list(files_count.values()))

    for key, val in files_count.items():
        if output["train"]["count_file"] <= int(total * train_ratio):
            output["train"]["projects"].append(key)
            output["train"]["count_file"] += val
        elif output["dev"]["count_file"] <= int(total * dev_ratio):
            output["dev"]["projects"].append(key)
            output["dev"]["count_file"] += val
        else:
            output["test"]["projects"].append(key)
            output["test"]["count_file"] += val

    return output


def merge_projects(projects_list, in_folder, out_file):
    write_file = open(out_file, "w", encoding="utf8")
    for project in tqdm(projects_list, desc=f"Writing data to: {out_file}"):
        with open(os.path.join(in_folder, project), 'r') as readFile:
            data = " ".join(readFile.readlines())
        data = data.replace("\n", " ")
        data = data.strip()
        if data:
            write_file.write(data + "\n")
    write_file.close()


parser = argparse.ArgumentParser(
    description='Split projects into train, dev and test ')

parser.add_argument('-i', '--in_folder', type=str,
                    default="data/raw_preprocessed/java", help='input folder location')
parser.add_argument('-o', '--out_folder', type=str,
                    default="data/splits/java", help='output folder location')
parser.add_argument('-t', '--split_type', type=str,
                    default="file", help='Options are (project, file, size)')
parser.add_argument('-r', '--split_ratio', type=str,
                    default="50-20-30", help='train-dev-test')
parser.add_argument('-s', '--seed', type=int, default=42)


def main():
    args = parser.parse_args()

    in_folder = args.in_folder
    out_folder = args.out_folder
    split_type = args.split_type

    if not os.path.exists(in_folder):
        print(f"Data dir does not exists: {in_folder}")
        exit(-1)

    try:
        os.makedirs(out_folder)
    except:
        pass

    projects = os.listdir(in_folder)

    random.seed(args.seed)
    random.shuffle(projects)

    projects_info = {}
    for project in projects:
        files_count = len(
            open(os.path.join(in_folder, project), "r").readlines())
        projects_info[project] = {
            "files_count": files_count,
            "size_in_bytes": os.path.getsize(os.path.join(in_folder, project))
        }

    output = split(projects_info, type=split_type, ratio=args.split_ratio)

    for key, items in output.items():
        merge_projects(
            projects_list=items['projects'], in_folder=in_folder,
            out_file=os.path.join(out_folder, f'{key}.txt')
        )


if __name__ == "__main__":
    main()
