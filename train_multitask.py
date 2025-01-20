"""
I want to classify sentences and also find all the entities as well as all the entity relations in a sentence using a bert model. Take the following sentences as example:

balance_enquiry:
I want to know my balance.
I want to know my savings account balance.
how much do I have in my current account?

fund_transfer:
I want to transfer money.
I want to send 500$ to Alif from my savings account.
I want to transfer $50 to my current account to my savings account.

Can you please provide a code for bert multitask fine-tuning to do this? use `bert-base-uncased` model
"""
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class MultitaskBERT(nn.Module):
    def __init__(self, model_name, num_intent_labels, num_ner_labels, num_re_labels):
        super(MultitaskBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.intent_head = nn.Linear(self.bert.config.hidden_size, num_intent_labels)
        self.ner_head = nn.Linear(self.bert.config.hidden_size, num_ner_labels)
        self.re_head = nn.Linear(self.bert.config.hidden_size, num_re_labels)

    def forward(self, input_ids, attention_mask, task, intent_labels=None, ner_labels=None, re_labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        pooled_output = outputs.pooler_output  # [batch_size, hidden_dim]

        if task == "intent":
            logits = self.intent_head(pooled_output)  # Intent classification
            loss = None
            if intent_labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, intent_labels)
            return logits, loss

        elif task == "ner":
            logits = self.ner_head(sequence_output)  # Token-level logits
            loss = None
            if ner_labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                # Flatten the logits and labels for loss computation
                loss = loss_fct(logits.view(-1, logits.size(-1)), ner_labels.view(-1))
            return logits, loss

        elif task == "re":
            logits = self.re_head(pooled_output)  # Relation classification
            loss = None
            if re_labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, re_labels)
            return logits, loss


class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=64):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(sentence, truncation=True, padding="max_length", max_length=self.max_length)
        return {
            "input_ids": torch.tensor(encoding.input_ids),
            "attention_mask": torch.tensor(encoding.attention_mask),
            "labels": torch.tensor(label)
        }


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=64):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(sentence, truncation=True, padding="max_length", max_length=self.max_length,
                                  is_split_into_words=True)
        word_ids = encoding.word_ids()

        # Align labels with tokenized output
        label_ids = [-100] * len(encoding.input_ids)  # -100 will be ignored in the loss
        for i, word_id in enumerate(word_ids):
            if word_id is not None and word_id < len(labels):
                label_ids[i] = labels[word_id]

        return {
            "input_ids": torch.tensor(encoding.input_ids),
            "attention_mask": torch.tensor(encoding.attention_mask),
            "labels": torch.tensor(label_ids)
        }


class REDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, entity_pairs, relations, tokenizer, max_length=64):
        self.sentences = sentences
        self.entity_pairs = entity_pairs
        self.relations = relations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.entity_pairs)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        entity_pair = self.entity_pairs[idx]
        relation = self.relations[idx]

        input_text = f"{sentence} [SEP] {entity_pair[0]} [SEP] {entity_pair[1]}"
        encoding = self.tokenizer(input_text, truncation=True, padding="max_length", max_length=self.max_length)

        return {
            "input_ids": torch.tensor(encoding.input_ids),
            "attention_mask": torch.tensor(encoding.attention_mask),
            "labels": torch.tensor(relation)
        }


if __name__ == '__main__':

    # Example intent data
    intent_sentences = [
        "I want to know my balance.",
        "I want to transfer money.",
        "how much do I have in my current account?",
        "I want to send 500$ to Alif from my savings account."
    ]
    intent_labels = [0, 1, 0, 1]  # 0: balance_enquiry, 1: fund_transfer

    # Initialize intent dataset
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    intent_dataset = IntentDataset(intent_sentences, intent_labels, tokenizer)



    # Example NER data
    ner_sentences = [
        ["I", "want", "to", "send", "$500", "to", "Alif", "from", "my", "savings", "account"],
        ["how", "much", "do", "I", "have", "in", "my", "current", "account"]
    ]
    ner_labels = [
        ["O", "O", "O", "O", "B-Amount", "O", "B-Person", "O", "O", "B-AccountType", "I-AccountType"],
        ["O", "O", "O", "O", "O", "O", "O", "B-AccountType", "I-AccountType"]
    ]

    # Map labels to integers
    # {0: 'O', 1: 'B-Amount', 2: 'B-Person', 3: 'B-AccountType', 4: 'I-AccountType'}
    ner_labels = [[0, 0, 0, 0, 1, 0, 2, 0, 0, 3, 4],
                  [0, 0, 0, 0, 0, 0, 0, 3, 4]]
    # Initialize NER dataset
    ner_dataset = NERDataset(ner_sentences, ner_labels, tokenizer)

    # Example RE data
    re_sentences = ["I want to send $500 to Alif from my savings account."]
    entity_pairs = [("$500", "Alif"), ("$500", "savings account"), ("Alif", "savings account")]
    relations = [0, 1, 2]  # 0: Amount-Sent-To, 1: Source-Account, 2: None

    # Initialize RE dataset
    re_dataset = REDataset(re_sentences * len(entity_pairs), entity_pairs, relations, tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torch.utils.data import DataLoader
    from transformers import AdamW
    from tqdm import tqdm
    import numpy as np

    # Hyperparameters
    batch_size = 16
    learning_rate = 5e-5
    num_epochs = 5

    # DataLoaders
    intent_dataloader = DataLoader(intent_dataset, batch_size=batch_size, shuffle=True)
    ner_dataloader = DataLoader(ner_dataset, batch_size=batch_size, shuffle=True)
    re_dataloader = DataLoader(re_dataset, batch_size=batch_size, shuffle=True)

    # Model and Optimizer
    model = MultitaskBERT(
        model_name="bert-base-uncased",
        num_intent_labels=2,  # 2 intents: balance_enquiry, fund_transfer
        num_ner_labels=5,  # 5 NER tags: O, B-Amount, B-Person, B-AccountType, I-AccountType
        num_re_labels=3  # 3 relations: Amount-Sent-To, Source-Account, None
    )
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Move model to device

    model.to(device)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Alternate between tasks
        for i, (intent_batch, ner_batch, re_batch) in enumerate(zip(intent_dataloader, ner_dataloader, re_dataloader)):
            # Intent Classification Task
            optimizer.zero_grad()
            intent_input_ids = intent_batch["input_ids"].to(device)
            intent_attention_mask = intent_batch["attention_mask"].to(device)
            intent_labels = intent_batch["labels"].to(device)
            _, intent_loss = model(
                intent_input_ids, intent_attention_mask, task="intent", intent_labels=intent_labels
            )
            intent_loss.backward()
            optimizer.step()

            # NER Task
            optimizer.zero_grad()
            ner_input_ids = ner_batch["input_ids"].to(device)
            ner_attention_mask = ner_batch["attention_mask"].to(device)
            ner_labels = ner_batch["labels"].to(device)
            _, ner_loss = model(ner_input_ids, ner_attention_mask, task="ner", ner_labels=ner_labels)
            ner_loss.backward()
            optimizer.step()

            # RE Task
            optimizer.zero_grad()
            re_input_ids = re_batch["input_ids"].to(device)
            re_attention_mask = re_batch["attention_mask"].to(device)
            re_labels = re_batch["labels"].to(device)
            _, re_loss = model(re_input_ids, re_attention_mask, task="re", re_labels=re_labels)
            re_loss.backward()
            optimizer.step()

            # Log losses
            if i % 10 == 0:  # Log every 10 steps
                print(
                    f"Step {i}: Intent Loss = {intent_loss.item():.4f}, "
                    f"NER Loss = {ner_loss.item():.4f}, RE Loss = {re_loss.item():.4f}"
                )

    torch.save(model.state_dict(), "multitask_bert_model.pth")
    print("Training complete.")

    model = MultitaskBERT(
        model_name="bert-base-uncased",
        num_intent_labels=2,
        num_ner_labels=5,
        num_re_labels=3
    )
    model.load_state_dict(torch.load("multitask_bert_model.pth"))
    model.to(device)
    # model.eval()
    print("Model loaded from multitask_bert_model.pth")

    def evaluate_intent(model, dataloader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits, _ = model(input_ids, attention_mask, task="intent")
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Intent Classification Accuracy: {accuracy:.4f}")
        return accuracy


    from sklearn.metrics import classification_report


    def evaluate_ner(model, dataloader, id2label):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits, _ = model(input_ids, attention_mask, task="ner")
                predictions = torch.argmax(logits, dim=2)
                for i in range(predictions.size(0)):
                    pred_labels = []
                    true_labels = []
                    for label_idx in range(len(labels[i])):
                        if labels[i][label_idx].item() != -100:
                            pred_labels.append(id2label[predictions[i][label_idx].item()])
                            true_labels.append(id2label[labels[i][label_idx].item()])
                    # pred_labels = [id2label[p.item()] for p in predictions[i] if p.item() != -100]
                    # true_labels = [id2label[l.item()] for l in labels[i] if l.item() != -100]
                    all_preds.extend(pred_labels)
                    all_labels.extend(true_labels)
        print("NER Classification Report:")
        print(classification_report(all_labels, all_preds))


    def evaluate_re(model, dataloader):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                logits, _ = model(input_ids, attention_mask, task="re")
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Relation Extraction Accuracy: {accuracy:.4f}")
        return accuracy



    # Evaluation
    intent_eval_dataloader = DataLoader(intent_dataset, batch_size=batch_size)
    ner_eval_dataloader = DataLoader(ner_dataset, batch_size=batch_size)
    re_eval_dataloader = DataLoader(re_dataset, batch_size=batch_size)

    # Intent Classification
    evaluate_intent(model, intent_eval_dataloader)

    # NER
    id2label = {0: 'O', 1: 'B-Amount', 2: 'B-Person', 3: 'B-AccountType', 4: 'I-AccountType'}
    evaluate_ner(model, ner_eval_dataloader, id2label)

    # Relation Extraction
    evaluate_re(model, re_eval_dataloader)

