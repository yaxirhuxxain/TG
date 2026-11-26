import os
import pickle
import re

import torch

from utils import all_equal

torch.cuda.synchronize()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def incremental_learning(data_files, context_size=10):
    data = []
    for file in data_files:
        with open(file, 'r') as rf:
            for file in rf.readlines():

                # split by [;{}] & filter out empty lines
                code_lines = re.findall('.*?[;{}]', file)
                code_lines = list(filter(None, code_lines)
                                  )  # remove none entries
                # Remove empty lines
                code_lines = [line.strip()
                              for line in code_lines if line.strip()]

                j = 0
                while (True):
                    if j >= len(code_lines) - 1:
                        break

                    # keep adding tokens until { token found to fix split between for loops
                    if code_lines[j].startswith("for ") and (code_lines[j].endswith(";")):
                        k = 1
                        try:
                            while not code_lines[j + k].endswith("{"):
                                if k >= 2:
                                    break
                                k += 1

                            # +1 becuase range takes one less the max length
                            tokens_stream = " ".join(
                                code_lines[j:j + k + 1]).split()
                            j = j + k
                        except:
                            tokens_stream = code_lines[j].split()
                    else:
                        tokens_stream = code_lines[j].split()

                    j += 1  # loop increment

                    # skip tokens with length less than 2 or having same values or grater than context_size
                    if len(tokens_stream) < 2 or all_equal(tokens_stream) or len(tokens_stream) > context_size:
                        continue

                    data.append(" ".join(tokens_stream))
    return data


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_files, context_size=10, type="WindowWise", word_idx=None, tokenizer=None, num_samples=None):
        """
        Args:
            data_files (string): Path to the file.
        """
        cache_dir = os.path.join(os.getcwd(), "cache", type)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        cache_file = f'{data_files[0].split("/")[-1].split(".")[0]}.pkl'
        cache_file_path = os.path.join(cache_dir, cache_file)

        if os.path.exists(cache_file_path):
            print(f"\t> Loading from Cache: {cache_file_path}")
            with open(cache_file_path, "rb") as fileHandler:
                self.data = pickle.load(fileHandler)

        else:
            if type == "WindowWise":
                self.data = []
                for file in data_files:
                    with open(file, 'r') as rf:
                        self.data.extend(rf.readlines())

            elif type == "Incremental":
                self.data = incremental_learning(data_files, context_size)
            else:
                print(f"Invalid type: {type} for preparing dataset")

            PAD = '<pad> ' * (context_size - 1)
            self.data = [PAD + item.replace('\n', '') for item in self.data]

            # tokenize text
            if tokenizer:
                self.data = [" ".join(tokenizer(item)) for item in self.data]

            # encoding data & preparing sequences
            context_size_with_pred = context_size + 1

            def encoder(item):
                return [word_idx[word] if word in word_idx else word_idx['<idf>'] for word in item.split()]

            temp = []
            for text in self.data:
                enc = encoder(text)
                for i in range(0, len(enc) - context_size_with_pred + 1):
                    temp.append(enc[i:i + context_size_with_pred])

            self.data = temp
            del temp  # empty temp data

            print(f"\t> Saving to Cache: {cache_file_path}")
            with open(cache_file_path, "wb") as fileHandler:
                pickle.dump(self.data, fileHandler,
                            protocol=pickle.HIGHEST_PROTOCOL)

        if num_samples:
            if num_samples <= 0 or num_samples > len(self.data):
                raise f"num_samples should be grater than zero and less than {len(self.data)}"
            self.data = self.data[0:num_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]
