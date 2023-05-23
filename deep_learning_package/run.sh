python3 main.py \
    --num_epochs 13 \
    --mode onTest \
    --model_type Cate \
    --train_batch_size 32 \
    --test_batch_size 32 \
    --save_logging_steps 500 \
    --learning_rate 5e-5 \
    --house_dim 1024\
    --beauty_dim 1024 \
    --baby_dim 256 \
    --kitchen_dim 256 \
    --data_path data/synthetic_final_data_train.csv \
    --test_path data/synthetic_final_data_test.csv \

python3 main.py \
    --num_epochs 13 \
    --mode onTest \
    --model_type Cate \
    --model_name distilroberta-base \
    --train_batch_size 32 \
    --test_batch_size 32 \
    --save_logging_steps 500 \
    --learning_rate 5e-5 \
    --house_dim 512 \
    --beauty_dim 256 \
    --baby_dim 256 \
    --kitchen_dim 256 \
    --data_path data/synthetic_final_data_train.csv \
    --test_path data/synthetic_final_data_test.csv \


python3 main.py \
    --num_epochs 13 \
    --mode onTest \
    --model_type Cate \
    --model_name climatebert/distilroberta-base-climate-f \
    --train_batch_size 32 \
    --test_batch_size 32 \
    --save_logging_steps 500 \
    --learning_rate 5e-5 \
    --house_dim 1024 \
    --beauty_dim 256 \
    --baby_dim 1024 \
    --kitchen_dim 256 \
    --data_path data/synthetic_final_data_train.csv \
    --test_path data/synthetic_final_data_test.csv \


python3 main.py \
    --num_epochs 13 \
    --mode onTest \
    --model_type noCate \
    --train_batch_size 32 \
    --test_batch_size 32 \
    --save_logging_steps 500 \
    --learning_rate 5e-5 \
    --house_dim 1024\
    --beauty_dim 1024 \
    --baby_dim 256 \
    --kitchen_dim 256 \
    --data_path data/synthetic_final_data_train.csv \
    --test_path data/synthetic_final_data_test.csv \


python3 main.py \
    --num_epochs 13 \
    --mode onTest \
    --model_type noCate \
    --model_name distilroberta-base \
    --train_batch_size 32 \
    --test_batch_size 32 \
    --save_logging_steps 500 \
    --learning_rate 5e-5 \
    --house_dim 512 \
    --beauty_dim 256 \
    --baby_dim 256 \
    --kitchen_dim 256 \
    --data_path data/synthetic_final_data_train.csv \
    --test_path data/synthetic_final_data_test.csv \


python3 main.py \
    --num_epochs 13 \
    --mode onTest \
    --model_type noCate \
    --model_name climatebert/distilroberta-base-climate-f \
    --train_batch_size 32 \
    --test_batch_size 32 \
    --save_logging_steps 500 \
    --learning_rate 5e-5 \
    --house_dim 1024 \
    --beauty_dim 256 \
    --baby_dim 1024 \
    --kitchen_dim 256 \
    --data_path data/synthetic_final_data_train.csv \
    --test_path data/synthetic_final_data_test.csv \

