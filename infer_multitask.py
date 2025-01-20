import torch
from transformers import AutoTokenizer
from train_multitask import MultitaskBERT


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = MultitaskBERT(
    model_name="bert-base-uncased",
    num_intent_labels=2,
    num_ner_labels=5,
    num_re_labels=3
)
model.load_state_dict(torch.load("multitask_bert_model.pth"))
# model.to(device)
# model.eval()
print("Model loaded from multitask_bert_model.pth")


def single_sentence_inference(model, sentence, tokenizer, label_maps):
    """
    Perform inference on a single sentence for all three tasks (Intent, NER, RE).

    Args:
    - model: MultitaskBERT model.
    - sentence: Input sentence (string).
    - tokenizer: Tokenizer for the model.
    - label_maps: A dictionary containing label mappings:
        - "intent_map": {id: intent_label}
        - "ner_map": {id: ner_label}
        - "re_map": {id: relation_label}

    Returns:
    - intent_result: Predicted intent label.
    - ner_result: List of (entity, entity_label) pairs.
    - re_result: List of (entity_pair, relation_label) pairs.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize the input sentence
    inputs = tokenizer(sentence, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Intent Classification
    with torch.no_grad():
        intent_logits, _ = model(input_ids, attention_mask, task="intent")
        intent_pred = torch.argmax(intent_logits, dim=1).item()
        intent_result = label_maps["intent_map"][intent_pred]

    # NER
    with torch.no_grad():
        ner_logits, _ = model(input_ids, attention_mask, task="ner")
        ner_preds = torch.argmax(ner_logits, dim=2).squeeze(0).tolist()
        ner_result = []
        tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

        for token, ner_pred in zip(tokens, ner_preds):
            if ner_pred != -100 and label_maps["ner_map"][ner_pred] != "O":
                ner_result.append((token, label_maps["ner_map"][ner_pred]))

    # Relation Extraction (RE)
    # For simplicity, assume entity pairs are manually specified or pre-detected
    # Example: Hardcoded entity pairs for demonstration
    entity_pairs = [
        ("$500", "Alif"),  # Example entity pairs
        ("$500", "savings account")
    ]
    re_result = []
    for entity1, entity2 in entity_pairs:
        input_text = f"{sentence} [SEP] {entity1} [SEP] {entity2}"
        re_inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        re_input_ids = re_inputs["input_ids"].to(device)
        re_attention_mask = re_inputs["attention_mask"].to(device)

        with torch.no_grad():
            re_logits, _ = model(re_input_ids, re_attention_mask, task="re")
            re_pred = torch.argmax(re_logits, dim=1).item()
            re_result.append(((entity1, entity2), label_maps["re_map"][re_pred]))

    return intent_result, ner_result, re_result



# Label Mappings
intent_map = {0: "balance_enquiry", 1: "fund_transfer"}
ner_map = {0: "O", 1: "B-Amount", 2: "B-Person", 3: "B-AccountType", 4: "I-AccountType"}
re_map = {0: "Amount-Sent-To", 1: "Source-Account", 2: "None"}

label_maps = {
    "intent_map": intent_map,
    "ner_map": ner_map,
    "re_map": re_map
}

# Example Sentence
sentence = "I want to send 500 taka to Alif from my savings account."

# Run Inference
intent_result, ner_result, re_result = single_sentence_inference(model, sentence, tokenizer, label_maps)

# Print Results
print(f"Intent: {intent_result}")
print(f"Entities: {ner_result}")
print(f"Relations: {re_result}")
