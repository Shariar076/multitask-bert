model_path = "bert-base-uncased-finetuned-ner"
sentence = "I want to send money to Alif from my savings account"


id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-NUM", 4: "I-NUM",
            5: "B-ACC", 6: "I-ACC", 7: "B-ORG", 8: "I-ORG"}  # Replace with your label mapping
import torch
from transformers import AutoTokenizer, BertForTokenClassification

# Load the fine-tuned model and tokenizer
# model_path = "path_to_your_finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertForTokenClassification.from_pretrained(model_path)

# Set the model to evaluation mode
model.eval()

# Define the NER labels mapping
# id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG"}  # Replace with your label mapping

# Input sentence
# sentence = "John Doe works at OpenAI."

# Tokenize the input sentence
inputs = tokenizer(sentence, return_tensors="pt", truncation=True, is_split_into_words=False)

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get the predicted label IDs
predicted_label_ids = torch.argmax(logits, dim=2).squeeze().tolist()

# Decode the tokens and their corresponding predicted NER labels
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())
predicted_labels = [id2label[label_id] for label_id in predicted_label_ids]

# Align tokens with original words
word_ids = inputs.word_ids()
aligned_labels = []
current_word = None
for idx, word_id in enumerate(word_ids):
    if word_id is None:
        continue
    if word_id != current_word:
        aligned_labels.append((tokens[word_id], predicted_labels[idx]))
        current_word = word_id

# Print the results
for word, label in aligned_labels:
    print(f"Word: {word}, Predicted NER Label: {label}")
