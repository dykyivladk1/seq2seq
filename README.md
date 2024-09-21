
# TransformerSeq2Seq Model for Text Summarization

![ADIFY AI Logo](assets/seq2seq.png)



This repository implements a Transformer-based sequence-to-sequence (Seq2Seq) model for text summarization. It leverages a positional encoding mechanism, multi-head attention, and feedforward layers for both encoding and decoding sequences. The dataset used in this project is the CNN/DailyMail dataset, which contains articles and their corresponding highlights.

## Project Structure

- `model.py`: Contains the TransformerSeq2Seq class and related submodules like positional encoding, multi-head attention, encoder, and decoder blocks.
- `dataset.py`: Implements the dataset class that processes the input articles and highlights.
- `train.py`: Script for training the model, containing the training loop, data processing, and tokenization steps.
- `generate_summary.py`: Script for generating summaries using a trained model.

## Sample Text Argument

The training script accepts a `--sample_text` argument, which specifies the number of sample articles to use during training (default is set to 30). This allows you to quickly test the model with a subset of the data.

## How to Run the Scripts

### Training the Model

To train the model using the `train.py` script, run the following command:

```bash
python train.py --data_path path/to/your/data.csv --batch_size 16 --epochs 30 --embed_size 512 --num_encoder_layers 6 --num_decoder_layers 6 --n_head 8 --max_len 512 --ff_hidden_mult 4 --dropout 0.1 --sample_text 30
```

- Replace `path/to/your/data.csv` with the actual path to the CSV file.
- You can adjust `batch_size`, `epochs`, and other parameters based on your requirements.

### Testing the Model (Generating Summaries)

To generate summaries using the `generate_summary.py` script, run the following command:

```bash
python generate_summary.py --article "Your article text here"
```

- Replace `"Your article text here"` with the actual text of the article you want to summarize.

## Requirements

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

Dependencies:
- `torch`: PyTorch library for deep learning models
- `einops`: For efficient tensor manipulations
- `transformers`: Huggingface transformers for tokenizer and pre-trained BART models
- `tqdm`: Progress bar for tracking training
- `pandas`: Data processing
