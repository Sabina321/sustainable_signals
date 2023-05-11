import argparse


parser = argparse.ArgumentParser(description='RecPlay')

################
# Train and Test set
################
parser.add_argument('--output_path', type=str, default="distilbert_finch_cat_tfidf_PR")
parser.add_argument('--data_path', type=str, default='data/final_data_train.csv')
parser.add_argument('--tfidf_path', type=str, default='data/tfidf.npy')
parser.add_argument('--test_path', type=str, default='data/final_data_test.csv')

parser.add_argument('--random_seed', type=int, default=45)
parser.add_argument('--data_split_seed', type=int, default=42)

################
# Training
################

parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--test_batch_size', type=int, default=16)
parser.add_argument('--save_total_limit', type=int, default=3)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--save_logging_steps', type=int, default=500)
parser.add_argument('--mode', type=str, default='CV')

###############
# Model
###############

parser.add_argument('--model_type', type=str, default='Anno_add')


parser.add_argument('--model_name', type=str, default='climatebert/distilroberta-base-climate-f')
#parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
parser.add_argument('--ro_model_name', type=str, default='roberta-base')
parser.add_argument("--lr_scheduler_type", default='linear', type=str,
            help="lr_scheduler_type")
parser.add_argument('--pretrain_ro_path', type=str, default='distilroberta-envclaim')
parser.add_argument('--pretrain_review', type=str, default='pre_review')


parser.add_argument('--house_dim', type=int, default=768)
parser.add_argument('--beauty_dim', type=int, default=768)
parser.add_argument('--baby_dim', type=int, default=768)
parser.add_argument('--kitchen_dim', type=int, default=768)

parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=5e-5, type=float,
            help="The initial learning rate for Adam.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
            help="Epsilon for Adam optimizer.")

parser.add_argument('--do_save', action='store_true')



################
args = parser.parse_args()