from transformers import BertTokenizer
from datasets import load_dataset
import tqdm

import nltk

#nltk.download('punkt')
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
a = load_dataset("wikipedia", "20220301.en")
print("start loading to memory ")
l = a.num_rows["train"]

ll = [0]
for i in range(16):
    ll.append(l // 16 * (i + 1))

ll_dis = []
for i in range(len(ll) - 1):
    ll_dis.append(a["train"]["text"][ll[i] : ll[i + 1]])

import re
def to_len(x):
    tokens = re.findall(r'\w+|[^\w\s]+', x)
    return len(tokens)

def crt_txt(i,  ll_dis):
    filename = "data/to" + str(i) + ".txt"
    l = len(ll_dis)
    s_len = 210
    char_list = [",", ".", "!", "?", ";", "\n"]
    with open(filename, "w", encoding="utf-8") as f:

        for row in tqdm.tqdm(range(l), desc="number of txt"):
          
            text1 = ll_dis[row]
            temp = nltk.tokenize.sent_tokenize(text1)
            
            temp2 = []
            for i in range(len(temp)):
                if to_len(temp[i]) <= s_len // 2:
                    temp2.append(temp[i])
                else:
                    s_t = temp[i]
                    for ii in range(len(s_t)):
                        if s_t[ii] in char_list and to_len(s_t[ii+1:]) <= s_len // 2:
                            temp2.append(s_t[:ii+1])
                            temp2.append(s_t[ii+1:])
                            break
            
           
            s_to = ""
            
            i = 0

            while i < len(temp2):
            
                while i < len(temp2) and temp2[i] == "\n":
                    i = i + 1

                if i < len(temp2):
                    s1 = temp2[i]  

                j = i + 1
            
                while j < len(temp2) and (to_len(s1) + to_len(temp2[j]) <= s_len // 2):
                    if temp2[j] != "\n" and temp2[j] != "":
                        s1 += " "
                        s1 += temp2[j]
                    j = j + 1
                
                s1 += " \t "

                if j >= len(temp2):
                    s2 = " \n "
                    s_to = s_to +  s1 + s2
                    break

                s2 = temp2[j]

                k = j + 1
                while k < len(temp2) and to_len(s2) + to_len(temp2[k]) <= s_len - to_len(s1):
                    if temp2[k] != "\n" :
                        s2 += " "
                        s2  += temp2[k]
                    k = k + 1

                s2 += " \n "

                s_to = s_to +  s1 + s2

                i = k
          
            f.write(s_to)
         
       


from threading import Thread 

thr_list = []
for i in range(16):
    mthread = Thread(target=crt_txt, args=(i, ll_dis[i]))
    mthread.start()
    thr_list.append(mthread)

for i in thr_list:
    i.join()

    

