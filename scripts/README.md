# Scripts Directory

This directory contains all model-specific scripts organized by model type.

## Directory Structure

- **llama3_1/**: Scripts for LLaMA 3.1 model
- **gemma3/**: Scripts for Gemma 3 model variants (1B, 4B, 12B, 27B)
- **ro_gpt2/**: Scripts for Romanian GPT-2 models (base, large)
- **ro_bert/**: Scripts for Romanian BERT models
- **rogemma2/**: Scripts for Romanian Gemma2 instruct DPO model
- **rollama3_1/**: Scripts for Romanian LLaMA 3.1 instruct DPO model
- **romistral_7b/**: Scripts for Romanian Mistral 7B instruct DPO model
- **generic/**: Generic scripts that can work with multiple models

## Usage

Each model directory contains:
- Training scripts (*.sh for fine-tuning)
- Inference scripts (*_inference.sh)

## Converting Models to GGUF Format

To convert models to GGUF format, use the `convert_hf_to_gguf.py` script from llama.cpp.

Example command to install the dependencies (from the root directory):

```bash
python3 -m venv .llamacpp-venv
source .llamacpp-venv/bin/activate
git clone https://github.com/ggml-org/llama.cpp.git tools/llama.cpp
cd tools/llama.cpp
pip install -r requirements.txt
```

Then, to convert a model, run:

```bash
python ./convert_hf_to_gguf.py <path_to_model_directory> --outfile <output_file.gguf> --outtype <output_type>
```

For example, to convert the Gemma 3 1B model:

```bash
python ./convert_hf_to_gguf.py ../../results/gemma3_1b_it_ft/ --outfile ../../results/gemma3_1b_it_ft/gemma.gguf --outtype f16
```

## Load Local LoUF Models in Ollama

First, create a Modfile like below:

```text
FROM <PATH_TO_MODEL.gguf>
```

For example, inside `./results/gemma3_1b_it_ft/`, where the `gemma.gguf` file is located, your Modfile should look like this:

```text
FROM gemma.gguf
```

Then, create the ollama model using the following command:

```bash
ollama create <model_name> --file <path_to_modfile>
```

For example, to create the Gemma 3 1B model:

```bash
ollama create gemma3_1b_it --file ./results/gemma3_1b_it_ft/Modfile
```

Finally, you can run the model.
