# Code for Thesis: A Non-Parametric Text Classification Approach Utilizing Lossless Compression Models

This paper is accepted to Findings of [ACL2023](https://aclanthology.org/2023.findings-acl.426/).

-------------------------

### Original Codebase

#### Require

Installation of Conda or Miniconda.

See `requirements.txt`.

Install requirements in a clean environment:

```sh
conda create -n npc python=3.7
conda activate npc
pip install -r requirements.txt
```

#### Run

```sh
python main_text.py
```

By default, this will only use 100 test and training samples per class as a quick demo. They can be changed by `--num_test`, `--num_train`.

```text
--compressor <gzip, lzw>
--dataset <AG_NEWS, DBpedia, YahooAnswers, 20News, R8, R52, kinnews, kirnews, swahili, filipino> [Note that for small datasets like kinnews, default 100-shot is too big, need to set --num_test and --num_train.]
--num_train <INT>
--num_test <INT>
--all_test [This will use the whole test dataset.]
--all_train [This will use the whole train dataset.]
--record [This will record the distance matrix in order to save for the future use. It's helpful when you when to run on the whole dataset.]
--test_idx_start <INT>
--test_idx_end <INT> [These two args help us to run on a certain range of test set. Also helpful for calculating the distance matrix on the whole dataset.]
--para [This will use multiprocessing to accelerate.]
--output_dir <DIR> [The output directory to save information of tested indices or distance matrix.]
```

#### Calculate Accuracy (Optional)

If we want to calculate accuracy from recorded distance file `<DISTANCE DIR>`, use

```sh
python main_text.py --record --score --distance_fn <DISTANCE DIR>
```

to calculate accuracy. Otherwise, the accuracy will be calculated automatically using the command in the last section.
