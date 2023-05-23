from datasets import load_dataset, Dataset, load_metric, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def compute_metrics(eval_preds):
    metric = load_metric('mse')
    logits, labels = eval_preds
    d = metric.compute(predictions=logits, references=labels)
    d["eval_Mse"] = d['mse']
    return d

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss (suppose one has 3 labels with different weights)
        loss = outputs.get("loss")
        #print(logits, labels)
        return (loss, outputs) if return_outputs else loss