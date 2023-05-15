# SUSTAINABLE SIGNALS: An AI Approach for Inferring Consumer Product Sustainability

This folder contains the code for the paper《SUSTAINABLE SIGNALS:
An AI Approach for Inferring Consumer Product Sustainability》


## Requirements

datasets==2.7.1

numpy==1.20.1

pandas==1.2.4

scikit_learn==1.0.2

scipy==1.6.2

torch==1.12.1

tqdm==4.59.0

transformers==4.25.1

We pre-trained several models. If you want to run the models other than ''Cate'', please download the folders 
''distilroberta-envclaim'' and ''pre_review'', and put them in the directory ''deep_learning_package''.


## Usage
The main implementation of SUSTAINABLE SIGNALS is in the folder ''deep_learning_package''

The hyperparameters for the SUSTAINABLE SIGNALS can be found in ''option.py''.

If you want to run the model, please make sure the working directory is in ''deep_learning_package'' and use the command:

```shell
python3 main.py \
    --num_epochs 13 \
    --mode onTest \
    --train_batch_size 32 \
    --test_batch_size 32 \
    --model_type Cate \
    --model_name distilbert-base-uncased \
    --save_logging_steps 500 \
    --learning_rate 5e-5 \
    --house_dim 768 \
    --beauty_dim 768 \
    --baby_dim 512 \
    --kitchen_dim 512
```

## Outputs

Update: You can find the outputs of our model in the "expe" folder.
