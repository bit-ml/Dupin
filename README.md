
# PAN 2020 Dataset preprocessing

Overview:
1. [Datasets summary](#Datasets-summary)
2. [PAN Closed-set setup](#Closed-set-setup)
    - [version v1](#-Version-v1)
    - [version v2](#-Version-v2)
3. [PAN Open-set setup](#Open-set-setup)
    - [unseen authors](#Unseen-authors-split)
    - [unseen fandoms](#Unseen-fandoms-split)
    - [unseen all](#Unseen-all-split)
4. [Datasets statistics](#Datasets-statistics)
5. [Original dataset files](#Original-dataset-files)
6. [Reddit datasets](#Reddit-datasets)

## Datasets summary
If you use these dataset splits, please cite both papers:
```
@article{DBLP:journals/corr/abs-2112-05125,
  author    = {Andrei Manolache and
               Florin Brad and
               Elena Burceanu and
               Antonio Barbalau and
               Radu Tudor Ionescu and
               Marius Popescu},
  title     = {Transferring BERT-like Transformers' Knowledge for Authorship Verification},
  journal   = {CoRR},
  volume    = {abs/2112.05125},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.05125},
  eprinttype = {arXiv},
  eprint    = {2112.05125},
  timestamp = {Mon, 13 Dec 2021 17:51:48 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-05125.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

```
@inproceedings{Kestemont2020OverviewOT,
  author    = {Mike Kestemont and
               Enrique Manjavacas and
               Ilia Markov and
               Janek Bevendorff and
               Matti Wiegmann and
               Efstathios Stamatatos and
               Martin Potthast and
               Benno Stein},
  editor    = {Linda Cappellato and
               Carsten Eickhoff and
               Nicola Ferro and
               Aur{\'{e}}lie N{\'{e}}v{\'{e}}ol},
  title     = {Overview of the Cross-Domain Authorship Verification Task at {PAN}
               2020},
  booktitle = {Working Notes of {CLEF} 2020 - Conference and Labs of the Evaluation
               Forum, Thessaloniki, Greece, September 22-25, 2020},
  series    = {{CEUR} Workshop Proceedings},
  volume    = {2696},
  publisher = {CEUR-WS.org},
  year      = {2020},
  url       = {http://ceur-ws.org/Vol-2696/paper\_264.pdf},
  timestamp = {Tue, 27 Oct 2020 17:12:48 +0100},
  biburl    = {https://dblp.org/rec/conf/clef/KestemontMMBWSP20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


| dataset |split type | filename | 
|---------|-----------|----------|
|PAN 2020|closed-set v1 and v2| [pan2020_closed_set_splits.zip](https://drive.google.com/file/d/18UPhYsdtFa8ObD0M6AeMdLxJ1XH42vHQ/view?usp=sharing) | 
|PAN 2020|open-set unseen authors | [pan2020_open_set_unseen_authors_splits.zip](https://drive.google.com/file/d/1F_NKLjHSpiPEviUC8B2h8zJkB9_Ftp8l/view?usp=sharing) | 
|PAN 2020|open-set unseen fandoms | [pan2020_open_set_unseen_fandoms_splits.zip](https://drive.google.com/file/d/1-GyRXROBhSKBtqJm9Hun5Acq4YmKKWL_/view?usp=sharing) | 
|PAN 2020|open-set unseen all | [pan2020_open_set_unseen_all.zip](https://drive.google.com/file/d/1Dl98Wxjej02DPqQSA3ofQ1Uk8eW6Uw9O/view?usp=sharing) | 
|reddit|open-set unseen authors| [minidarkreddit_authorship_verification.zip](https://drive.google.com/file/d/1ok_CY59RhD0GgJqF1OOZMN592Zp9fgOY/view?usp=sharing) | 



## Closed-set setup
In the closed-set setup authors of same-author pairs in the validation/test set are guaranteed to appear in the training set. However, this is difficult to achieve for the different-author pairs of the PAN 2020 dataset, as they span a large number of authors with few occurences each.

### Files
Download [```pan2020_closed_set_splits.zip```](https://drive.google.com/file/d/18UPhYsdtFa8ObD0M6AeMdLxJ1XH42vHQ/view?usp=sharing) and unzip it. This is the structure of its content: 
```
xl/
   v1_split/
            pan20-av-large-test.jsonl
            pan20-av-large-notest.jsonl
   v2_split/
            pan20-av-large-test.jsonl
            pan20-av-large-notest.jsonl
xs/
   v1_split/
            pan20-av-small-test.jsonl
            pan20-av-small-notest.jsonl
   v2_split/
            pan20-av-small-test.jsonl
            pan20-av-small-notest.jsonl
```
We try two variants of splitting the datasets, called ```v1``` and ```v2```. The splits for the PAN 2020 large dataset can be found in the ```xl``` folder, while the splits for the PAN 2020 small dataset can be found in the ```xs``` folder.


### Version v1
In this version, authors of same-author pairs in the validation set are guaranteed to appear in the training set, while some authors of different-author pairs in the validation set may not appear in the training set.

Here are some dataset statistics:
|dataset | filename | size | SA / SA-SF / SA-DF | DA / DA-SF / DA-DF | 
|--------|----------|------|----|----|
|PAN 2020 large - original| ```pan20-av-large.jsonl``` | 275565 | - | - | -|
|PAN 2020 large - test | ```pan20-av-large-test.jsonl``` | 13784 | 7395/0/7395 | 6389/1114/5275 |
|PAN 2020 large - w/o test| ```pan20-av-large-no-test.jsonl``` | 261784 | - | - | -|
|PAN 2020 large - train | ```pan20-av-large-train.jsonl``` | 248688 | 133359/0/133359 | 115329/20945/94384 |
|PAN 2020 large - val | ```pan20-av-large-val.jsonl``` | 13090 | 7024/0/7024 | 6069/1072/4997 |  
where:
 - SA: same-author pairs
 - SA-SF: same-author pairs that have the same fandom
 - SA-DF: same-author pairs that have different fandoms
 - DA: different-author pairs
 - DA-SF: different-author pairs that have the same fandom
 - DA-DF: different-author pairs that have different fandoms


To split the PAN 2020 large dataset (```pan20-av-large-notest.jsonl```) into train (```pan20-av-large-train.jsonl```) and validation (```pan20-av-large-val.jsonl```) splits using the ```v1``` version, call the ```split_jsonl_dataset``` function in ```preprocess/split_train_val.py```:
```
cd preprocess
python split_train_val.py
```
Make sure to specify the correct paths:
 - ```path_to_train_jsonl``` is where you want to save your training split
 - ```path_to_test_jsonl``` is where you want to save your validation split
```
    # split Train dataset into Train and Val
    split_jsonl_dataset(
        path_to_original_jsonl=pan20-av-notest.jsonl,
        path_to_train_jsonl=pan20-av-large-train.jsonl,
        path_to_test_jsonl=pan20-av-large-val.jsonl,
        split_function=split_pan_dataset_closed_set_v1,
        test_split_percentage=0.05
    )
```


Different-author pairs:
 - the DA pairs are randomly assigned to train/val split
 - unseen authors can appear at evaluation, for instance (A1, A2) in training set and (A3, A4) in val set. 

 Same-author (SA) pairs:
 - while populating the validation split, SA pairs are evenly assigned to train/val splits
 - for instance, if we have 10 SA examples from a given author, we assign 5 examples to training split and 5 examples to validation split. This ensures that the author of SA pairs in the validation split has been 'seen' at training time*. 
 - *this may result in unseen fandoms at validation time though, for instance (A1, F1, A1, F2) at training time and (A1, F3, A1, F4) at validation 

 

### Version v2
If we separate the DA pairs (ai, aj) into two groups Train and Test, such that both authors (ai, aj) of DA pairs in Test also appear in DA pairs in Train
(or SA pairs), we get the following stats:
 - Number of DA pairs:  127787
 - Number of candidate test pairs:  181
 - Number of candidate train pairs:  127606

The small number of candidate test pairs suggest that most of the authors in the DA pairs of the test split are 'unseen' at training
time. To loosen this restriction, we can split the DA pairs such that at least one of the authors (ai, aj) in a DA Test pair appears in
other DA Train pairs or in SA pairs. We get the following stats:
 - Number of DA pairs:  127787
 - Number of candidate test pairs:  17894
 - Number of candidate train pairs:  109893

We therefore split a PAN dataset into Train and Val/Test such that at least one of the authors in DA Test pairs appears DA train pairs or SA train pairs.
The SA pairs of an author A are equally distributed between Train and Test.

Here are some dataset statistics:
|dataset | filename | size | SA / SA-SF / SA-DF | DA / DA-SF / DA-DF | 
|--------|----------|------|----|----|
|PAN 2020 large - original| ```pan20-av-large.jsonl``` | 275565 | - | - | -|
|PAN 2020 large - test | ```pan20-av-large-test.jsonl``` | 13785 | 7396/0/7396 | 6389/355/6034 |
|PAN 2020 large - w/o test| ```pan20-av-large-no-test.jsonl``` | 261784 | - | - | -|
|PAN 2020 large - train | ```pan20-av-large-train.jsonl``` | 248688 | 133359/0/133359 | 115329/22420/92909 |
|PAN 2020 large - val | ```pan20-av-large-val.jsonl``` | 13090 | 7023/0/7023 | 6069/356/5713 |  

To split the PAN 2020 large dataset (```pan20-av-large-notest.jsonl```) into train and validation splits using the ```v2``` version, call the ```split_jsonl_dataset``` function in ```preprocess/split_train_val.py```:
```
cd preprocess
python split_train_val.py
```
Make sure to specify the correct paths and split function:
```
    # split Train dataset into Train and Val
    split_jsonl_dataset(
        path_to_original_jsonl=pan20-av-notest.jsonl,
        path_to_train_jsonl=pan20-av-large-train.jsonl,
        path_to_test_jsonl=pan20-av-large-val,
        split_function=split_pan_dataset_closed_set_v2,
        test_split_percentage=0.05
    )
```


## Open-set setup
In the open-set setup, authors and fandoms in the test set do not appear in the training set. This is difficult to achieve simultanously, so we have create 2 splits: unseen authors and unseen fandoms.
### Unseen authors split
In this split, authors in the test set do not appear in the training set. However, this is difficult to achieve for the PAN 2020 dataset, so we split it into train and val/test sets such that: 
 - authors of same-author (SA) pairs in the test set do not appear in SA training pairs 
 - some authors (<5%) of different-author (DA) pairs in the test set may appear in the DA training pairs
 - most of the fandoms in the test set appear in the training set 


### Files
Download [```pan2020_open_set_unseen_authors_splits.zip```](https://drive.google.com/file/d/1F_NKLjHSpiPEviUC8B2h8zJkB9_Ftp8l/view?usp=sharing) and unzip it. This is the structure of its content: 
```
unseen_authors/
    xl/
        pan20-av-large-test.jsonl
        pan20-av-large-notest.jsonl
    xs/
        pan20-av-small-test.jsonl
        pan20-av-small-notest.jsonl
```


Here are some dataset statistics:
|dataset | filename | size | SA / SA-SF / SA-DF | DA / DA-SF / DA-DF | 
|--------|----------|------|----|----|
|PAN 2020 large - original| ```pan20-av-large.jsonl``` | 275565 | - | - | -|
|PAN 2020 large - test | ```pan20-av-large-test.jsonl``` | 13777 | 7388/0/7388 | 6389/2061/4328 |
|PAN 2020 large - w/o test| ```pan20-av-large-no-test.jsonl``` | 261788 | 140390/0/140390 | 121398/21070/100328 |
|PAN 2020 large - train | ```pan20-av-large-train.jsonl``` | 248699 | 133367/0/133367 | 115332/18840/96492 |
|PAN 2020 large - val | ```pan20-av-large-val.jsonl``` | 13089 | 7023/0/7023 | 6066/2230/3836 |  

To split the PAN 2020 large dataset (```pan20-av-large-notest.jsonl```) into train and validation splits, call the ```split_jsonl_dataset``` function in ```preprocess/split_train_val.py```:
```
cd preprocess
python split_train_val.py
```
Make sure to specify the correct paths and split function:
```
    # split Train dataset into Train and Val
    split_jsonl_dataset(
        path_to_original_jsonl=pan20-av-notest.jsonl,
        path_to_train_jsonl=pan20-av-large-train.jsonl,
        path_to_test_jsonl=pan20-av-large-val,
        split_function=split_pan_dataset_open_set_unseen_authors,
        test_split_percentage=0.05
    )
```
### Unseen fandoms split
In this split type:
 - examples at test/val time belong to fandoms that have not been seen during training
- some authors in the val/test set may also appear in the train set
- training examples (d1, d2, f1, f2) where either f1 or f2 appear in the test fandoms are dropped => this results in ~110K fewer training examples

All train/val/test splits are provided.

### Files
Download [```pan2020_open_set_unseen_fandoms_splits.zip```](https://drive.google.com/file/d/1-GyRXROBhSKBtqJm9Hun5Acq4YmKKWL_/view?usp=sharing) and unzip it. This is the structure of its content: 
```
unseen_fandoms/
    xl/
        pan20-av-large-train.jsonl
        pan20-av-large-val.jsonl
        pan20-av-large-test.jsonl
    xs/
        pan20-av-small-train.jsonl
        pan20-av-small-val.jsonl
        pan20-av-small-test.jsonl
```

Here are some XL dataset statistics:
|dataset | filename | size | SA / SA-SF / SA-DF | DA / DA-SF / DA-DF | 
|--------|----------|------|----|----|
|PAN 2020 XL - original| ```pan20-av-large.jsonl``` | 275565 | 147778/0/147778 | 127787/23131/104656 | 
|PAN 2020 XL - train | ```pan20-av-large-train.jsonl``` | 133990 | 71826/0/71826 | 62164/20779/41385 |
|PAN 2020 XL - val | ```pan20-av-large-val.jsonl``` | 13451 | 7047/0/7047 | 6408/1176/5232 |
|PAN 2020 XL - test | ```pan20-av-large-test.jsonl``` | 13453 | 7056/0/7056 | 6409/1176/5233 |


Here are some XS dataset statistics:
|dataset | filename | size | SA / SA-SF / SA-DF | DA / DA-SF / DA-DF |
|--------|----------|------|----|----|
|PAN 2020 XS - original| ```pan20-av-small.jsonl``` | 52601 | 27834/0/27834 | 24767/0/24767 | 
|PAN 2020 XS - train | ```pan20-av-small-train.jsonl``` | 36859 | 22547/0/22547 | 14312/0/14312 |
|PAN 2020 XS - val | ```pan20-av-small-val.jsonl``` | 4179 | 2568/0/2568 | 1393/0/1393 | 
|PAN 2020 XS - test | ```pan20-av-small-test.jsonl``` | 4180 | 2719/0/2719 | 1394/0/1394 |

To split the PAN 2020 original dataset (```pan20-av-*.jsonl```) into train/validation/test splits, call the ```split_jsonl_dataset_into_train_val_test``` function in ```preprocess/split_train_val.py```:
```
cd preprocess
python split_train_val.py
```

For the XS dataset:
```
    split_jsonl_dataset_into_train_val_test(
        path_to_original_jsonl=paths_dict['original'],
        path_to_train_jsonl=paths_dict['train'],
        path_to_val_jsonl=paths_dict['val'],
        path_to_test_jsonl=paths_dict['test'],
        split_function=split_pan_small_dataset_open_set_unseen_fandoms,
        test_split_percentage=0.2
    )
```
For the XL dataset:
```
    split_jsonl_dataset_into_train_val_test(
        path_to_original_jsonl=paths_dict['original'],
        path_to_train_jsonl=paths_dict['train'],
        path_to_val_jsonl=paths_dict['val'],
        path_to_test_jsonl=paths_dict['test'],
        split_function=split_pan_dataset_open_set_unseen_fandoms,
        test_split_percentage=0.1
    )
```
### Unseen all split
In this split type:
 - authors and fandoms in the test set have not been seen
 in the training data
- authors in validation set have not been seen in the training set, but validation fandoms are similar to the training fandoms

All train/val/test splits are provided.

### Files
Download [```pan2020_open_set_unseen_all_splits.zip```](TODO) and unzip it. This is the structure of its content: 
```
unseen_all/
    xl/
        pan20-av-large-train.jsonl
        pan20-av-large-val.jsonl
        pan20-av-large-test.jsonl
    xs/
        pan20-av-small-train.jsonl
        pan20-av-small-val.jsonl
        pan20-av-small-test.jsonl
```

Here are some XL dataset statistics:
|dataset | filename | size | SA / SA-SF / SA-DF | DA / DA-SF / DA-DF | 
|--------|----------|------|----|----|
|PAN 2020 XL - original| ```pan20-av-large.jsonl``` | 275565 | 147778/0/147778 | 127787/23131/104656 | 
|PAN 2020 XL - train | ```pan20-av-large-train.jsonl``` | 248001 | 124000/0/124000 | 124001/62286/61715 |
|PAN 2020 XL - val | ```pan20-av-large-val.jsonl``` | 13703 | 6852/0/6852 | 6851/2966/3885 |
|PAN 2020 XL - test | ```pan20-av-large-test.jsonl``` | 13704 | 6853/0/6853 | 6851/1633/5218 |


Here are some XS dataset statistics:
|dataset | filename | size | SA / SA-SF / SA-DF | DA / DA-SF / DA-DF |
|--------|----------|------|----|----|
|PAN 2020 XS - original| ```pan20-av-small.jsonl``` | 52601 | 27834/0/27834 | 24767/0/24767 | 
|PAN 2020 XS - train | ```pan20-av-small-train.jsonl``` | 36851 | 18425/0/18425 | 18426/31/18395 |
|PAN 2020 XS - val | ```pan20-av-small-val.jsonl``` | 4003 | 2002/0/2002 | 2001/2/1999 | 
|PAN 2020 XS - test | ```pan20-av-small-test.jsonl``` | 4001 | 2000/0/2000 | 2001/3/1998 |

To split the PAN 2020 original dataset (```pan20-av-*.jsonl```) into train/validation/test splits, call the ```split_jsonl_dataset_resampling``` function in ```preprocess/split_train_val.py```:
```
cd preprocess
python split_train_val.py
```

For the XL dataset:
```
train_examples, val_examples, test_examples = split_jsonl_dataset_resampling(
        path_to_original_jsonl=None,
        path_to_authors_json=paths_dict['authors_dict'],
        path_to_train_jsonl=paths_dict['train'],
        path_to_val_jsonl=paths_dict['val'],
        path_to_test_jsonl=paths_dict['test'],
        train_size=248000,
        test_size=13700
)
```



## Datasets statistics
### PAN 2020 large dataset (XL)
The PAN 2020 large dataset has 275.565 examples, detailed here:
|  | same fandom | cross-fandom |
|--|-------------|--------------|
|same-author pairs| 0 | 147.778 |
|different-author pairs| 23.131 | 104.656 |

 - same-author pairs are constructed from 41.370 authors, while different-author pairs are constructed from 251.503 authors
 - 14.704 authors in SA pairs can be found in DA pairs as well
 - 3.966 authors in DA pairs appear in at least one DA pair
 - author tuples (Ai, Aj) in DA pairs are unique (i.e. authors 532 and 7145 can be found in this combination only once in DA pairs)
 - there are 494.236 distinct documents

We now detail the closed-set and open-set setups. In both setups, we split the XL dataset into 95% training and 5% test and the XS dataset into 90% training and 10% test. 

### PAN 2020 small dataset (XS):
The PAN 2020 small dataset has 52.601 examples, detailed here:
|  | same fandom | cross-fandom |
|--|-------------|--------------|
|same-author pairs| 0 | 27.834 |
|different-author pairs| 0 | 24.767 |

## Original dataset files

| dataset | original examples file | original ground truth file | merged file |
|---------|---------------|-------------------|-------------|
|PAN 2020 XS | ```pan20-authorship-verification-training-small.jsonl```|```pan20-authorship-verification-training-small-truth.jsonl```|```pan20-av-small.jsonl```|
|PAN 2020 XL | ```pan20-authorship-verification-training-large.jsonl```|```pan20-authorship-verification-training-large-truth.jsonl```|```pan20-av-large.jsonl```|


 We concatenate the original data and ground truth files into a single file ```pan20-av-*.jsonl``` by calling the ```merge_data_and_labels()``` function.


 Since the ```.jsonl``` files are quite large, we use the ```write_jsonl_to_folder()``` function to store examples from ```pan20-av-*.jsonl``` into separate ```.json``` files inside a folder.
 
 ## Reddit datasets

 |dataset | train/val/test sizes
|---------|---------------------|
|reddit closed set| 284/486/558 |
|reddit open set (unseen authors)| 204/412/412|

## Models 
TODO
