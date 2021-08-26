# **TDD Team- Mukayese: An Extensive Benchmark for Turkish NLP**
Mukayese is a benchmarking platform based on [EvalAI](https://github.com/Cloud-CV/EvalAI) project for various Turkish NLP tools and tasks, ranging from Spell-checking to Natural Language Understanding tasks (NLU).

## How to Use
PhD Magic...


## Goals
The most important goal of Mukayese is to standardize the comparison and evaulation of Turkish natural language processing methods. As a result of the lack of a platform for benchmarking, Turkish Natural Language Processing researchers struggle with comparing their models to the existing ones due to several problems that we solve:
1. Not all  datasets in the literature have specified train/validation/test splits or the test sets are annotated. This results in a point where the reported results in a publication must be double-checked by the researcher reviewing the literature to ensure that the eevaluation is made with the same method that researcher uses. Furthermore, not all reported performance values are to be correct and might have been corrupted by the (probably unintentional) mistakes of the researcher. We solve this problem by evaluating the models with datasets from different distributions in which the annotations of test splits are not publicized. To ensure fairness in leaderboard listings, we evaluate the models with open source scripts and disclosed specified versions and settings of the used libraries.  
1. In many papers, authors do not include open source implementations of their works. This prevents the researchers to analyse the models and geting a greater understanding of the proposed method. Moreover, when unpublished, these models cannot be used for purposes fine-tuning or retraining with a different set of hyperparameters. We address this problem by labeling the submissions with which an open source implementation provided "verified". As the TDD Team, we test the submitted open source implementation, review it from the unbiased perspective of different researchers and require it to be published in an easy-to-use manner.   
1. Benchmarking systems like [GLUE](https://gluebenchmark.com/) and [SuperGLUE](https://super.gluebenchmark.com/) provide a way for researchers to test a model they developed on an extensive set of tasks. We aim to do a better job with 

## General Overview

## Benchmarks
Currently, we provide leaderboards in 8 different tasks and on X different datasets
1. Spell-checking and Correction - Custom Dataset
2. Text Classification - [OffensEval](https://sites.google.com/site/offensevalsharedtask/multilingual)
3. Language Modeling - [trwiki-67](https://data.tdd.ai/#/6bdc4da6-7638-4adc-825b-d101918439bb) and [trnews-64](https://github.com/tdd-ai/trnews-64)
4. Named-Entity Recognition (arda)
5. Machine Translation - [OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles2018.php) and [MUST-C](https://ict.fbk.eu/must-c/)
6. Natural Language Inference - [XNLI](https://github.com/facebookresearch/XNLI)
7. Tokenization - Custom Datasets
8. Part-of-speech Tagging - [UD-Turkish-BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN)

## Datasets
Under this project, we created 5 distinct datasets with in-depth documentation and train/validation/test splits for two datasets. In addition, all the datasets presented by our team in [Turkish Data Depository](https://data.tdd.ai/#/) are published.

#### Datasets Created under this Project
1. [trwiki-67](https://github.com/tdd-ai/trwiki-67)
2. [trnews-64](https://github.com/tdd-ai/trnews-64)
3. [asd]()
4. [asd]()

#### Datasets for which Train/Test/Validation Splits are Generated
1. NER (arda'nın referans koyması gerekiyor)
2. [TrMor2018](https://github.com/ai-ku/TrMor2018)

## Trained Baseline Models
For baseline models to start the leaderboards with, we trained 18 distinct models for 8 different tasks. All of the scripts of the  pretrained models and respective details may be found [in this repository we created](https://github.com/tdd-ai/mukayese-baselines). 

#### Spell-checking and Correction
As the TDD team, we developed a state-of-the-art [Hunspell](https://github.com/hunspell/hunspell)-based [spell-checker](https://github.com/tdd-ai/spell-checking-and-correction) that is reported alongside comparsisons of performance of 7 different models:  [TurkishSpellChecker](https://github.com/StarlangSoftware/TurkishSpellChecker-Py), [zemberek-nlp](https://github.com/ahmetaa/zemberek-nlp), [zemberek-python](https://github.com/Loodos/zemberek-python), [velhasil](https://github.com/MiniVelhasil/velhasil), [hunspell-tr](https://github.com/vdemir/hunspell-tr) (vdemir), [hunspell-tr](https://github.com/hrzafer/hunspell-tr) (hrzafer), [tr-spell](https://code.google.com/archive/p/tr-spell/).

#### Text Classification
[PhD magic...](https://github.com/alisafaya/OffensEval2020)

#### Language Modelling
PhD Magic...
#### Named-Entity Recognition
Senior year magic
#### Machine Translation
For machine translation, we trained [Fairseq](https://github.com/facebookresearch/fairseq), [NPMT](https://github.com/posenhuang/NPMT), [Tensor2tensor](https://github.com/tensorflow/tensor2tensor#translation) models on the Turkish-English subsets of 2 different datasets: [OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles2018.php) and [MUST-C](https://ict.fbk.eu/must-c/).

#### Natural Language Inference
Senior year magic
#### Tokenization

#### Part-of-speech Tagging
Senior year magic

## Future Directions 
In this section, the future plans of our project are listed. 
#### Turkish Natural Language Processing Competitions
In addition to the challenges that are always open to submissions, we plan to organise Turkish Natural Language Processsing challenges and allow researchers to submit their ideas for contests which, after approval from our team, will be hosted. 
#### New Benchmarks
We plan to present the following benchmarks, on which we have started to work, in the future:
1. Morphological Analysis - [Trmor2018](https://github.com/ai-ku/TrMor2018)
1. Document Classification - [TTC-4900](https://huggingface.co/datasets/ttc4900#dataset-card-for-ttc4900-a-benchmark-data-for-turkish-text-categorization), [1150 News](https://data.tdd.ai/#/d2fe5fc8-2d2f-4fde-aad6-5e4b0dd1c1db) and [Trt-11](https://github.com/gurkan08/datasets/tree/master/trt_11_category)
1. Question Answering - [XQuad](https://github.com/deepmind/xquad) and [TQuad](https://github.com/TQuad/turkish-nlp-qa-dataset)
2. Summarization - PhD Magic needed...
3. Reading Comprehension - PhD Magic needed...
4. Dependency Parsing - [UD Turkish BOUN](https://github.com/UniversalDependencies/UD_Turkish-BOUN)

#### A Turkish Natural Langauge Processing Library
Since we require the open source implementation for submissions, we plan to create a library with the submitted models and their data loaders, tokenizers etc. that will be widely used by the Turkish Natural Language Processing researchers. The core idea is to gather as many Turkish NLP models as possible in a single library where they can be imported in a few lines of code. 

## Team Members
Ali Safaya - @asafaya19
Emirhan Kurtuluş - @ekurtulus
Arda Göktoğan - @ardofski
Our Mentor - Dorukhan Afacan @dafajon
Our Advisor - Prof. Deniz Yüret @dyuret
