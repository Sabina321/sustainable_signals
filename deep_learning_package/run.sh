for mt in climatebert/distilroberta-base-climate-f 
do
    for hou_dim in 512 768
    do
        for bea_dim in 512 768
        do
            for bb_dim in 512 768
            do
                for kit_dim in 512 768
                do

                python3 main.py \
                        --num_epochs 13 \
                        --mode onTest \
                        --train_batch_size 32 \
                        --test_batch_size 32 \
                        --model_type Cate \
                        --model_name $mt \
                        --save_logging_steps 500 \
                        --learning_rate 5e-5 \
                        --house_dim $hou_dim \
                        --beauty_dim $bea_dim \
                        --baby_dim $bb_dim \
                        --kitchen_dim $kit_dim \

                done
            done
        done
    done
done



