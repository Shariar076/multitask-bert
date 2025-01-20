from datasets import load_dataset, Dataset, Value, Features, Sequence, ClassLabel, DatasetDict
from sympy import false
import pandas as pd
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

model_id = "bert-base-uncased"
# model_id = "distilbert-base-uncased"
# model_id = "ProsusAI/finbert"
task = "ner"

# dataset_id="conll2003"
# # dataset_id = "DFKI-SLT/cross_ner"
# dataset = load_dataset(dataset_id, trust_remote_code=True)
#



df = pd.read_csv("Dataset - nltk_tokenizer.csv")[['tokens', 'ner_tags']]



df["tokens"] = df["tokens"].apply(eval)
df["ner_tags"] = df["ner_tags"].apply(eval)

df = df[df['ner_tags'].apply(lambda li: any(x!=0 for x in li))]

train_df, eval_df = train_test_split(df, train_size=.9)

ner_labels = ["O", "B-PER", "I-PER", "B-NUM", "I-NUM", "B-ACC", "I-ACC", "B-ORG", "I-ORG"]
features = Features({'tokens': Sequence(Value("string")), 'ner_tags': Sequence(ClassLabel(names=ner_labels))})
train_dataset = Dataset.from_pandas(train_df).remove_columns("__index_level_0__").cast(features)
val_dataset = Dataset.from_pandas(eval_df).remove_columns("__index_level_0__").cast(features)

dataset = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

print(dataset['train'])
print(dataset["train"].features[f"ner_tags"])

# dataset['train'].to_pandas().to_csv("conll2003.csv", index=False)

label_list = dataset["train"].features[f"{task}_tags"].feature.names
print(label_list)

tokenizer = AutoTokenizer.from_pretrained(model_id)
# print(tokenizer.tokenize(' '.join(['that' 'is' 'to' 'end' 'the' 'state' 'of' 'hostility' ',' '"' 'Thursday'
#  "'s" 'overseas' 'edition' 'of' 'the' 'People' "'s" 'Daily' 'quoted'
#  'Tang' 'as' 'saying' '.'])))

label_all_tokens = True
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
for example in tokenized_datasets['train']:
    print(example)
'''
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model = AutoModelForTokenClassification.from_pretrained(model_id, num_labels=len(label_list))



model_name = model_id.split("/")[-1]
args = TrainingArguments(
    f"{model_name}-finetuned-{task}",
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    push_to_hub=False,
)



from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer)

example = dataset["train"][4]
# print(example["tokens"])
#
# tokenized_input = tokenizer(example["tokens"], is_split_into_words=True)
# tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
# print(tokens)
from evaluate import load
metric = load("seqeval")
# labels = [label_list[i] for i in example[f"{task}_tags"]]
# results = metric.compute(predictions=[labels], references=[labels])
#
# print({
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     })

import numpy as np
from seqeval.metrics import classification_report
from seqeval.metrics import accuracy_score

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }



trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()


print(trainer.evaluate())
predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])



# Convert predictions to NER labels
# id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG"}  # Replace with your label mapping

def decode_predictions(predictions, labels):
    decoded_results = []
    for pred_row, label_row in zip(predictions, labels):
        decoded_row = []
        for pred_id, label_id in zip(pred_row, label_row):
            if label_id != -100:  # Ignore special tokens
                decoded_row.append((ner_labels[pred_id],
                                    ner_labels[label_id]))
        decoded_results.append(decoded_row)
    return decoded_results

# Post-process predictions (argmax over logits)
predictions = predictions.argmax(axis=-1)

# Decode results
decoded_results = decode_predictions(predictions, labels)

# Print results
for i, row in enumerate(decoded_results):
    print(f"Row {i}:")
    for pred, true in row:
        print(f"  Predicted: {pred}, True: {true}")
'''