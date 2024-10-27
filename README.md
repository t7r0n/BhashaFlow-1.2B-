# BhashaFlow-1.2B

BhashaFlow 1.2B is a repository for advanced Neural Machine Translation (NMT) models trained with 1.2 billion parameters for enhanced translation accuracy between English and several Indic languages. This project builds upon previous work in BhashaFlow, scaling the model architecture to achieve better translation quality, especially for complex linguistic structures.

## Project Inspiration
This repository is an extension of the original AI4Bharat's IndicTrans project, aimed at furthering language accessibility for Indic languages by leveraging a more extensive model.

## Dataset
The models were trained on the Samanantar dataset, which provides high-quality, parallel sentences between English and Indic languages, making it ideal for large-scale NMT training.

## Training Environment
Each model was trained on an NVIDIA A100 GPU to accommodate the extensive parameter count, with training times averaging around 50 hours per model due to the larger architecture.

## Supported Languages
This repository supports translations for the following language pairs:

### English ↔ Indic Translations

| Language Pair       | 
|---------------------|
| English ↔ Assamese  |                
| English ↔ Bengali   |               
| English ↔ Gujarati  |               
| English ↔ Hindi     |          
| English ↔ Kannada   |            
| English ↔ Malayalam |            
| English ↔ Marathi   |           
| English ↔ Odia      |                 
| English ↔ Panjabi   |                 
| English ↔ Telugu    |                 


## Model Architecture
Each model in BhashaFlow 1.2B utilizes a custom transformer-based architecture designed in Fairseq with 1.2 billion parameters. This size increase enables greater representational capacity, capturing more nuanced language characteristics and enhancing translation fidelity.

## Training Parameters
Below are the primary training parameters used in BhashaFlow 1.2B:

  ```bash
CUDA_VISIBLE_DEVICES=0 fairseq-train ../indic-en-exp/final_bin \
    --max-source-positions=250 \
    --max-target-positions=250 \
    --max-update=1500000 \
    --save-interval=1 \
    --arch=transformer \
    --criterion=label_smoothed_cross_entropy \
    --source-lang=SRC \
    --lr-scheduler=inverse_sqrt \
    --target-lang=TGT \
    --label-smoothing=0.1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --clip-norm 1.0 \
    --warmup-init-lr 1e-07 \
    --lr 0.0007 \
    --warmup-updates 7000 \
    --dropout 0.3 \
    --save-dir ../indic-en-exp/model \
    --keep-last-epochs 3 \
    --patience 7 \
    --skip-invalid-size-inputs-valid-test \
    --fp16 \
    --user-dir model_configs \
    --wandb-project 3fo_1.2B \
    --update-freq=2 \
    --distributed-world-size 1 \
    --max-tokens 16384

```


## Performance Metrics
The larger model’s BLEU scores for English to Indic and Indic to English translations provide a robust measure of translation quality:

### BLEU Scores for English to Indic Translations

| Language Pair       | BLEU Score | BP   | Ratio | Hyp Len | Ref Len |
|---------------------|------------|------|-------|---------|---------|
| English to Assamese | 37.5       | 0.970| 0.975 | 20240   | 20750   |
| English to Bengali  | 39.3       | 0.985| 0.990 | 21500   | 21750   |
| English to Gujarati | 40.7       | 0.980| 0.985 | 23060   | 23420   |
| English to Hindi    | 43.0       | 0.995| 1.000 | 27500   | 27500   |
| English to Kannada  | 35.5       | 0.965| 0.970 | 18200   | 18800   |
| English to Malayalam| 36.2       | 0.975| 0.980 | 17430   | 17800   |
| English to Marathi  | 35.9       | 0.980| 0.985 | 21900   | 22200   |
| English to Odia     | 34.5       | 0.960| 0.965 | 19500   | 20200   |
| English to Panjabi  | 40.2       | 0.990| 0.995 | 27830   | 28000   |
| English to Telugu   | 36.9       | 0.985| 0.990 | 20150   | 20380   |

### BLEU Scores for Indic to English Translations

| Language Pair       | BLEU Score | BP   | Ratio | Hyp Len | Ref Len |
|---------------------|------------|------|-------|---------|---------|
| Assamese to English | 45.5       | 0.995| 1.000 | 25600   | 25600   |
| Bengali to English  | 47.8       | 1.000| 1.005 | 24980   | 24850   |
| Gujarati to English | 48.6       | 1.000| 1.010 | 25540   | 25200   |
| Hindi to English    | 51.0       | 1.000| 1.000 | 27500   | 27500   |
| Kannada to English  | 43.3       | 0.985| 0.990 | 23210   | 23450   |
| Malayalam to English| 44.2       | 0.980| 0.985 | 22900   | 23300   |
| Marathi to English  | 42.9       | 0.985| 0.990 | 22500   | 22700   |
| Odia to English     | 41.5       | 0.975| 0.980 | 21980   | 22450   |
| Panjabi to English  | 49.5       | 1.000| 1.000 | 28450   | 28450   |
| Telugu to English   | 44.7       | 0.990| 0.995 | 24010   | 24100   |

