import pandas as pd

import json
from scipy import stats

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DistilBertForSequenceClassification, AutoConfig, AutoModel, AutoModelForSequenceClassification

from options import args
from preprocess import preprocess, preprocess_cv
from model import noCate, Cate, Env_add, Anno_add, Anno_only
from trainer import MyTrainer, compute_metrics

MODEL_DICT = {"noCate": noCate,
			  "Cate": Cate,
			  "Env_add": Env_add,
			  "Anno_only": Anno_only,
			  "Anno_add": Anno_add}

def CV(args):
	res = []
	for fold in range(5):

		tokenized_dataset, data_collator, tfidf_size, tokenizer, ro_tokenzier = preprocess_cv(args, fold)
		config = AutoConfig.from_pretrained(args.model_name, num_labels=1)
		model = MODEL_DICT[args.model_type](config=config, args=args)

		training_args = TrainingArguments(
			f"{args.output_path}",
			save_total_limit=args.save_total_limit,
			num_train_epochs=args.num_epochs,
			evaluation_strategy="steps",
			learning_rate=args.learning_rate,
			save_steps=args.save_logging_steps,
			logging_steps=args.save_logging_steps,
			fp16=True,
			metric_for_best_model='Mse',
			greater_is_better=False,
			lr_scheduler_type=args.lr_scheduler_type,
			# load_best_model_at_end=True,
			per_device_train_batch_size=args.train_batch_size,
			per_device_eval_batch_size=args.test_batch_size,
		)

		trainer = MyTrainer(
			model=model,
			args=training_args,
			train_dataset=tokenized_dataset["train"],
			eval_dataset=tokenized_dataset["test"],
			tokenizer=tokenizer,
			# data_collator=data_collator,
			compute_metrics=compute_metrics
			# callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
		)

		trainer.train()
		eva = trainer.evaluate(tokenized_dataset["test"])

		pred = trainer.predict(tokenized_dataset["test"])
		sem = stats.sem((pred[0].reshape(-1) - pred[1])**2, axis=None, ddof=0)

		res.append(eva)
		res.append(sem)

	expe_name = "expe/"
	if args.model_name == "climatebert/distilroberta-base-climate-f":
		expe_name += "climatebert"
	elif args.model_name == "distilbert-base-uncased":
		expe_name += "distilbert"
	else:
		expe_name += "roberta"
	expe_name += "_" + str(args.random_seed)
	expe_name += "_" + str(args.mode)
	expe_name += "_" + str(args.model_type)
	expe_name += "_" + str(args.house_dim)
	expe_name += "_" + str(args.beauty_dim)
	expe_name += "_" + str(args.baby_dim)
	expe_name += "_" + str(args.kitchen_dim)
	expe_name += "_" + str(args.learning_rate)
	expe_name += ".json"

	with open(expe_name, "w") as f:
		json.dump(res, f)

	return res

def onTest(args):

	tokenized_dataset, data_collator, tfidf_size, tokenizer, ro_tokenzier = preprocess(args)
	config = AutoConfig.from_pretrained(args.model_name, num_labels=1)
	model = MODEL_DICT[args.model_type](config=config, args=args)

	training_args = TrainingArguments(
		f"{args.output_path}",
		save_total_limit=args.save_total_limit,
		num_train_epochs=args.num_epochs,
		evaluation_strategy="steps",
		learning_rate=args.learning_rate,
		save_steps=args.save_logging_steps,
		logging_steps=args.save_logging_steps,
		fp16=True,
		metric_for_best_model='Mse',
		greater_is_better=False,
		lr_scheduler_type=args.lr_scheduler_type,
		# load_best_model_at_end=True,
		per_device_train_batch_size=args.train_batch_size,
		per_device_eval_batch_size=args.test_batch_size,
	)

	trainer = MyTrainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_dataset["train"],
		eval_dataset=tokenized_dataset["test"],
		tokenizer=tokenizer,
		#data_collator=data_collator,
		compute_metrics=compute_metrics
		# callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
	)

	trainer.train()
	eva = trainer.evaluate(tokenized_dataset["test"])
	pred = trainer.predict(tokenized_dataset["test"])
	sem = stats.sem(pred[0].reshape(-1) - pred[1], axis=None, ddof=0)

	df = pd.DataFrame(list(zip(pred[0].reshape(-1), pred[1])), columns=["prediction", "label"])



	expe_name = "expe/"
	expe_name += "_" + str(args.random_seed)
	if args.model_name == "climatebert/distilroberta-base-climate-f":
		expe_name += "_" + "climatebert"
	elif args.model_name == "distilbert-base-uncased":
		expe_name += "_" + "distilbert"
	else:
		expe_name += "_" + "roberta"
	expe_name += "_" + str(args.mode)
	expe_name += "_" + str(args.model_type)
	expe_name += "_" + str(args.house_dim)
	expe_name += "_" + str(args.beauty_dim)
	expe_name += "_" + str(args.baby_dim)
	expe_name += "_" + str(args.kitchen_dim)
	expe_name += "_" + str(args.learning_rate)

	df.to_csv(expe_name + ".csv", index=False)
	expe_name += ".json"

	res = [eva, sem]
	with open(expe_name, "w") as f:
		json.dump(res, f)

	return eva

def ploting(args):

	tokenized_dataset, data_collator, tfidf_size, tokenizer, ro_tokenzier = preprocess_cv(args, 0)
	config = AutoConfig.from_pretrained(args.model_name, num_labels=1)
	model = MODEL_DICT[args.model_type](config=config, args=args)

	training_args = TrainingArguments(
		f"{args.output_path}",
		save_total_limit=args.save_total_limit,
		num_train_epochs=args.num_epochs,
		evaluation_strategy="steps",
		learning_rate=args.learning_rate,
		save_steps=args.save_logging_steps,
		logging_steps=args.save_logging_steps,
		fp16=True,
		metric_for_best_model='Mse',
		greater_is_better=False,
		lr_scheduler_type=args.lr_scheduler_type,
		# load_best_model_at_end=True,
		per_device_train_batch_size=args.train_batch_size,
		per_device_eval_batch_size=args.test_batch_size,
	)

	trainer = MyTrainer(
		model=model,
		args=training_args,
		train_dataset=tokenized_dataset["train"],
		eval_dataset=tokenized_dataset["test"],
		tokenizer=tokenizer,
		#data_collator=data_collator,
		compute_metrics=compute_metrics
		# callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
	)

	trainer.train()
	eva = trainer.evaluate(tokenized_dataset["test"])



	expe_name = "expe/"
	expe_name += "_" + str(args.random_seed)
	expe_name += "_" + str(args.mode)
	expe_name += "_" + str(args.model_type)
	expe_name += "_" + str(args.lr_scheduler_type)
	expe_name += "_" + str(args.learning_rate)
	expe_name += "_" + str(args.num_epochs)
	expe_name += ".json"



	res = trainer.state.log_history + [eva]
	with open(expe_name, "w") as f:
		json.dump(res, f)

	return eva

def main(args):
	if args.mode == "CV":
		eva = CV(args)
	elif args.mode == "ploting":
		eva = ploting(args)
	else:
		eva = onTest(args)

if __name__ == "__main__":
	predictions = main(args)


	# if args.do_save:
	# 	model.save_pretrained(args.save_path)
	# 	tokenizer.save_pretrained(args.save_path)
	#
	# # python transformer_models.py --do_save