# SeLeRoSa - Sentence-Level Romanian Satire Detection Dataset

[![DOI](https://zenodo.org/badge/1003740539.svg)](https://doi.org/10.5281/zenodo.15689793)

## Abstract

> Satire, irony, and sarcasm are techniques that can disseminate untruthful yet plausible information in the news and on social media, akin to fake news. These techniques can be applied at a more granular level, allowing satirical information to be incorporated into news articles. In this paper, we introduce the first sentence-level dataset for Romanian satire detection for news articles, called SeLeRoSa. The dataset comprises 13,873 manually annotated sentences spanning various domains, including social issues, IT, science, and movies. With the rise and recent progress of large language models (LLMs) in the natural language processing literature, LLMs have demonstrated enhanced capabilities to tackle various tasks in zero-shot settings. We evaluate multiple baseline models based on LLMs in both zero-shot and fine-tuning settings, as well as baseline transformer-based models, such as Romanian BERT. Our findings reveal the current limitations of these models in the sentence-level satire detection task, paving the way for new research directions.

## Description 

This repository contains the code and data for the paper "SeLeRoSa - Sentence-Level Romanian Satire Detection Dataset". This dataset is intended for the task of satire detection in Romanian news articles at the sentence level.

The dataset is provided in two versions:
* [Anonymized](./data/csv/selerosa.csv): this is a lightly processed version of the dataset
* [Processed](./data/csv/selerosa_proc.csv): this is processed version of the dataset, which includes additional text processing steps such as lemmatization, stop word removal, and punctuation removal.

For more details about the anonymization and processing of the dataset, please refer to the [Dataset](#dataset) section below.

The dataset can also be found on [Hugging Face](https://huggingface.co/collections/unstpb-nlp/selerosa-sentence-level-romanian-satire-detection-dataset-6852b46fa93704e84b05a7a9).

## Dataset

The anonymization stage involved using Spacy to replace named entities with tokens:
  * Persons → `<PERSON>`
  * Nationalities/religious/political groups → `<NORP>`
  * Geopolitical entities → `<GPE>`
  * Organizations → `<ORG>`
  * Locations → `<LOC>`
  * Facilities → `<FAC>`

In addition, we replace URLs with the `@URL` tag. Manual filtering of frequently occurring entities not caught by NER was also performed.

The processing step involved:

 * Convert text to lowercase
 * Fix diacritics according to Romanian standards
 * "ţ" and "ş" (cedilla) vs "ț" and "ș" (comma)
 * Tokenize the text using Spacy's tokenizer
 * Remove stop words
 * Remove punctuation marks
 * Remove short tokens (less than 3 characters)
 * Lemmatize text using Spacy's Romanian lemmatizer

For the unprocessed dataset (anonymized only), check https://huggingface.co/datasets/unstpb-nlp/SeLeRoSa
For the processed dataset, check https://huggingface.co/datasets/unstpb-nlp/SeLeRoSa-proc

## Dataset Usage

You can use the dataset directly from the Hugging Face Datasets library, or use the csv file from this repository.

### Using the Hugging Face Datasets Library

First, install the following dependencies:

```bash
pip install datasets torch
```

Example of loading the train set in a dataloader:

```python
from datasets import load_dataset
from torch.utils.data import DataLoader

dataset = load_dataset("unstpb-nlp/SeLeRoSa", split="train")
# Or to use the processed version:
# dataset = load_dataset("unstpb-nlp/SeLeRoSa", split="train")

dataloader = DataLoader(dataset)
for sample in dataloader:
    print(sample)
```

### Using the CSV file

The csv files can be found in the `data/csv/` directory of this repository. You can load them using pandas:

```python
import pandas as pd

df = pd.read_csv("data/csv/selerosa.csv")
# or for the processed version
# df = pd.read_csv("data/csv/selerosa_proc.csv")

print(df.head())
```

Then, the dataset can be loaded into a huggingface dataset as follows:

```python
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import pandas as pd

# Load your CSV file
df = pd.read_csv('data/csv/selerosa.csv') # or 'data/csv/selerosa_proc.csv' for the processed version

# Split the dataset based on the 'split' column and reset index
train_df = df[df['split'] == 'train'].drop('split', axis=1).reset_index(drop=True)
val_df = df[df['split'] == 'val'].drop('split', axis=1).reset_index(drop=True)
test_df = df[df['split'] == 'test'].drop('split', axis=1).reset_index(drop=True)

# Create DatasetDict
dataset_dict = DatasetDict({
    'train': Dataset.from_pandas(train_df),
    'validation': Dataset.from_pandas(val_df),
    'test': Dataset.from_pandas(test_df)
})

# Create a DataLoader for the training set
dataloader = DataLoader(dataset_dict["train"])
for sample in dataloader:
    print(sample)
```

## Dataset structure

The following columns are available for every sample:

| Field | Data Type | Description |
|---|---|---|
| index | int | A unique identifier for every sentence |
| news_id | int | A unique identifier for the source news associated with the current sentence |
| sentence | string | The anonymized sentence |
| domain | string | The domain associated with the sentence. Can be one of: `life-death`, `it-stiinta`, `cronica-de-film` |
| label_0 | int | The label given by the first annotator. 0 - regular, 1 - satirical |
| label_1 | int | The label given by the second annotator 0 - regular, 1 - satirical |
| label_2 | int | The label given by the third annotator 0 - regular, 1 - satirical |
| label | int | The aggregated label through majority voting. This should be used for training and evaluation. 0 - regular, 1 - satirical |
| split | string | (only for the CSV files) The split associated with the sentence. Can be one of: `train`, `validation`, `test` |

## Statistics

Some statistics about the dataset can be found in the `results/figures/` directory.

## Experiments

In the `satire/experiments/` directory, you can find the code used to run the experiments presented in the paper. There are scripts for training and evaluating the models, which can be found in the `scripts/` directory.

To run the experiments, you will need to install the following dependencies:

```bash
pip install -r requirements.txt
```

Then, you can run the training and evaluation scripts as follows:

```bash
./satire/gpt4/gpt4o.sh
```

> [!NOTE]
> You need to set your OpenAI API key in the `OPENAI_API_KEY` environment variable before running the script.
> In addition, to log on wandb, you need to set the `WANDB_API_KEY` environment variable.

> [!NOTE]
> To run the local LLMs experiments, you need to install the NVIDIA CUDA dependencies:
> 
> `pip install -r requirements_nvidia.txt`
>
> The inference will use vllm to run the local LLMs.

> [!NOTE]
> For Gemma 3 1B, we saw some serious performance issues under vllm, so we recommend using ollama instead. To convert models to ollama format, check [`./scripts/README.md`](./scripts/README.md).

## Citation
If you use this dataset in your research, please cite as follows:

```bibtex
@software{smadu_2025_15689794,
  author       = {Smădu, Răzvan-Alexandru and
                  Iuga, Andreea and
                  Cercel, Dumitru-Clementin and
                  Pop, Florin},
  title        = {SeLeRoSa - Sentence-Level Romanian Satire
                   Detection Dataset
                  },
  month        = jun,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.15689794},
  url          = {https://doi.org/10.5281/zenodo.15689794},
  swhid        = {swh:1:dir:5c0e7a00a415d4346ee7a11f18c814ef3c3f5d88
                   ;origin=https://doi.org/10.5281/zenodo.15689793;vi
                   sit=swh:1:snp:e8bb60f04bd1b01d5e3ac68d7db37a3d28ab
                   7a22;anchor=swh:1:rel:ff1b46be53b410c9696b39aa7f24
                   a3bd387be547;path=razvanalex-phd-SeLeRoSa-83693df
                  },
}
```

## License

The code is released under the [MIT](./LICENSE) License. The dataset is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) License.