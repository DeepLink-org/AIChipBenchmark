from torch.utils.data import Dataset
import tqdm
import torch
import random
import numpy as np
from transformers import BertTokenizer
import os
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")



class BERTDataset(Dataset):
    def __init__(self, corpus_path, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.lines = []
        
        print(corpus_path)
        file_names = os.listdir(corpus_path)
        for file in file_names:
            files = corpus_path + "/" + file
            print(files)
            with open(files, "r", encoding=encoding) as f:                
                self.lines.extend([line[:-1].split("\t")
                                for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines) if line.find("\t") != -1])
                self.corpus_lines = len(self.lines)


    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)

        t1_encode = tokenizer.encode(t1)
        t1_random, t1_label = self.random_word(t1_encode)

        t2_encode = tokenizer.encode(t2)
        t2_random, t2_label = self.random_word(t2_encode)
        t2_random = t2_random[1:]
        t2_label = t2_label[1:]
         
        segment_label = ([1 for _ in range(len(t1_random))] + [2 for _ in range(len(t2_random))])[:self.seq_len]
        
        bert_input = (t1_random + t2_random)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]

        padding = [tokenizer.pad_token_id for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, tokens):
        
        output_label = [0 for _ in range(len(tokens))]
        tar_mask = int((len(tokens) - 2) * 0.15)
        now_mask = 0
        lth = len(tokens) - 1
        for i in range(1, lth):
            if tokens[i] == tokenizer.mask_token_id:
                continue
            if now_mask >= tar_mask:
                break

            prob = random.random()
            if prob < 0.15:
                output_label[i] = tokens[i]
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    temp = tokens[i]
                    tokens[i] = tokenizer.mask_token_id
                    now_mask += 1
                    # if tokenizer.convert_ids_to_tokens(temp).startswith("##"):
                    #     j = i - 1
                    #     while j >= 0 and tokenizer.convert_ids_to_tokens(tokens[j]).startswith("##"):
                    #         output_label[j] = tokens[j]
                    #         tokens[j] = tokenizer.mask_token_id
                    #         j -= 1
                    #         now_mask += 1
                    #     if j:
                    #         output_label[j] = tokens[j]
                    #         tokens[j] = tokenizer.mask_token_id
                    #         now_mask += 1

                    #     j = i + 1
                    #     while j < lth and tokenizer.convert_ids_to_tokens(tokens[j]).startswith("##"):
                    #         output_label[j] = tokens[j]
                    #         tokens[j] = tokenizer.mask_token_id
                    #         now_mask += 1
                    #         j += 1

                    # elif i + 1 < lth and tokenizer.convert_ids_to_tokens(tokens[i + 1]).startswith("##"):
                    #     j = i + 1
                    #     while j < lth and tokenizer.convert_ids_to_tokens(tokens[j]).startswith("##"):
                    #         output_label[j] = tokens[j]
                    #         tokens[j] = tokenizer.mask_token_id
                    #         j += 1
                    #         now_mask += 1
                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randint(0, 30521)
                    
                # 10% randomly change token to current token
                else:
                    pass

            else:
                pass
             

        return tokens, output_label

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):

        return self.lines[item][0], self.lines[item][1]
    

    def get_random_line(self):
        
        return self.lines[random.randrange(len(self.lines))][1]
