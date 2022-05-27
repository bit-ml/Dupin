import json
from tqdm import tqdm
import os
import re

#     if 'RC' in path_string and 'bz2' not in path_string:
#         file_path = path_string
#         darknet_output = path_string + '_darknet'

#         f_in = open(file_path, 'r')
#         f_out = open(darknet_output, 'a')

#         for idx, line in tqdm(enumerate(f_in)):
#             line = json.loads(line)
#             sub_name = line['subreddit'].lower()
#             if sub_name == 'dnmuk' or sub_name == 'darknetmarkets':
#                 json.dump(line, f_out, indent=2)

def convert_to_jsonl(json_path: str):
    with open(json_path) as fp:
        x = json.load(fp)
    
    jsonl_path = json_path + 'l'
    total_tokens = 0
    if os.path.exists(jsonl_path):
        print("file %s already exists, deleting it" % (jsonl_path))
        os.remove(jsonl_path)
    with open(jsonl_path, 'w') as fout:
        for idx, line in enumerate(tqdm(x)):
            c1, c2 = line['pair'][0], line['pair'][1]
            c12 = c1 + ' ' + c2
            num_tokens = len(re.split("\s+", c12.strip(), flags=re.UNICODE))
            total_tokens += num_tokens
            json.dump(line, fout)
            fout.write('\n')

    avg_tokens = total_tokens / (2*len(x))

    print("avg num tokens for %s = %d" % (jsonl_path, avg_tokens))
    

if __name__ == '__main__':
    SR1_PATHS = {
        'train': '/darkweb/darknet_authorship_verification/silkroad1/darknet_authorship_verification_train.json',
        'val': '/darkweb/darknet_authorship_verification/silkroad1/darknet_authorship_verification_val.json',
        'test': '/darkweb/darknet_authorship_verification/silkroad1/darknet_authorship_verification_test.json'
    }
    AGORA_PATHS = {
        'train': '/darkweb/darknet_authorship_verification/agora/darknet_authorship_verification_train.json',
        'val': '/darkweb/darknet_authorship_verification/agora/darknet_authorship_verification_val.json',
        'test': '/darkweb/darknet_authorship_verification/agora/darknet_authorship_verification_test.json'
    }
    DARKREDDIT_PATHS = {
        'train': '/darkweb2/darkreddit_authorship_verification/darkreddit_authorship_verification_train.json',
        'val': '/darkweb2/darkreddit_authorship_verification/darkreddit_authorship_verification_val.json',
        'test': '/darkweb2/darkreddit_authorship_verification/darkreddit_authorship_verification_test.json'
    }
    
    # for path_dict in [SR1_PATHS, AGORA_PATHS, DARKREDDIT_PATHS]:
    #     for split in ['train', 'val', 'test']:
    #         convert_to_jsonl(path_dict[split])
            #print(path_dict[split])
    dir_path = "/pan2020/reddit_darknet/reddit_open_split/train"
    pan2020_path = "/pan2020/pan20-authorship-verification-training-small/pan20-authorship-verification-training-small.jsonl"
    total_tokens = 0
    num_files = 0
    with open(pan2020_path) as fp:
        for idx, line in enumerate(tqdm(fp)):
            line = json.loads(line)
            c1, c2 = line['pair'][0], line['pair'][1]
            c12 = c1 + ' ' + c2
            num_tokens = len(re.split("\s+", c12.strip(), flags=re.UNICODE))
            total_tokens += num_tokens
            num_files += 1

    avg_tokens = total_tokens / (2 * num_files)
    print("Avg tokens = ", avg_tokens)
    
    # total_tokens = 0
    # for fname in os.listdir(dir_path):
    #     full_path = os.path.join(dir_path, fname)
    #     with open(full_path) as fp:
    #         line = json.load(fp)
    #         c1, c2 = line['pair'][0], line['pair'][1]
    #         c12 = c1 + ' ' + c2
    #         num_tokens = len(re.split("\s+", c12.strip(), flags=re.UNICODE))
    #         total_tokens += num_tokens
    #num_files = len(os.listdir(dir_path))
    # avg_tokens = total_tokens / (2 * num_files)
    # print("Avg tokens = ", avg_tokens)

    #convert_to_jsonl(DARKREDDIT_PATHS['test'])

    # json_files = []
    # with open(DARKREDDIT_PATHS['test']+'l') as fp:
    #     for idx, line in enumerate(fp.readlines()):
    #         print(line)
    #         json_files.append(json.loads(line))