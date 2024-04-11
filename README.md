# MLX RAG

Retrieval Augmented Generation (RAG) example using [Langchain](https://python.langchain.com/docs/get_started/quickstart/) (loader), [Chroma](https://docs.trychroma.com/getting-started) (vector database), [Hugging Face sentence-transformers](https://huggingface.co/sentence-transformers) (embeddings) and [MLX](https://huggingface.co/mlx-community) (LLM).

This example code was tested with these models:

- `mlx-community/gemma-1.1-7b-it-4bit`
- `mlx-community/Mistral-7B-Instruct-v0.2-4bit`
- `mlx-community/Mixtral-8x7B-Instruct-v0.1-hf-4bit-mlx`

## Usage

```
pip install -r requirements.txt
python main.py
```

![mlx-rag](https://gist.github.com/assets/7937/65e44579-3d29-49b7-a49d-aff51e741fba)

### Example document

- [Chemical strategies to mitigate electrostatic charging during coffee grinding](https://arxiv.org/abs/2312.03103) (Access Paper -> View PDF)
