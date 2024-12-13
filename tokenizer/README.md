To decide on what vocab size to use I trained tokenizers of various sizes with [train_tokenizer.py](train_tokenizer.py) and used [test_tokenizer.py](test_tokenizer.py) to compute their compression ratio (total characters / total tokens) on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) training set:

| vocab size | compression ratio |
|------------|-------|
| 32768      | 4.213 |
| 16384      | 4.207 |
| 8192       | 4.177 |
| 4096       | 4.014 |
| 2048       | 3.665 |
| 1024       | 3.066 |

So seeing minimal dropoff I will use a small vocabulary of 4096.