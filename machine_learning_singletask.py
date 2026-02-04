import torch
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import torch.nn as nn
import torch.optim as optim
import pandas as pd

def labelize(labels, id_to_label=[]):
    label_to_id = dict()
    for i, label in enumerate(id_to_label):
        label_to_id[label] = i

    for label in labels:
        if label not in label_to_id:
            label_to_id[label] = len(id_to_label)
            id_to_label.append(label)
    
    labelsi = [label_to_id[label] for label in labels]

    return (labelsi, id_to_label)



class DataSet:
    def __init__(self, sentences, labels, labels_to_text):
        self.sentences = sentences
        self.labels = labels
        self.labels_to_text = labels_to_text
    
    def get_label_text(self, i):
        return self.labels_to_text[i]



class Data:
    def __init__(self):
        self.train_claim = pd.read_csv("data/checkworthy/english_train.tsv", sep='\t')
        self.dev_claim = pd.read_csv("data/checkworthy/english_dev.tsv", sep='\t')
        self.devtest_claim = pd.read_csv("data/checkworthy/english_dev-test.tsv", sep='\t')

        self.train_stance = pd.read_csv("data/stance/cleaned_train.tsv", sep='\t')
        self.dev_stance = pd.read_csv("data/stance/cleaned_dev.tsv", sep='\t')

        self.claim_train_set = self.get_train_claim()
        self.claim_dev_set = self.get_dev_claim(self.claim_train_set.labels_to_text)
        self.claim_devtest_set = self.get_devtest_claim(self.claim_dev_set.labels_to_text)
        self.stance_train_set = self.get_train_stance()
        self.stance_dev_set = self.get_dev_stance(self.stance_train_set.labels_to_text)
    
    def get_claim(self, df, labels2=[]):
        sentences = list(df["Text"])
        (labels, labels_to_text) = labelize(df["class_label"], labels2)
        return DataSet(sentences, labels, labels_to_text)

    def get_train_claim(self, labels=[]):
        return self.get_claim(self.train_claim, labels)

    def get_dev_claim(self, labels=[]):
        return self.get_claim(self.dev_claim, labels)

    def get_devtest_claim(self, labels=[]):
        return self.get_claim(self.devtest_claim, labels)

    def get_stance(self, df, labels2=[]):
        sentences = list(df["rumor"])
        (labels, labels_to_text) = labelize(df["label"], labels2)
        return DataSet(sentences, labels, labels_to_text)

    def get_train_stance(self, labels=[]):
        return self.get_stance(self.train_stance, labels)

    def get_dev_stance(self, labels=[]):
        return self.get_stance(self.dev_stance, labels)



class SingleTaskBERT(nn.Module):
    def __init__(self, num_labels):
        super(SingleTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Task-specific layers
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output) # zero out 10% of the output at random
        logits = self.classifier(pooled_output)
        
        return logits



class TrainingSet:
    def __init__(self, data_set, tokenizer):
        self.size = len(data_set.labels_to_text)

        # Tokenize and prepare datasets separately for each task
        self.inputs = tokenizer(data_set.sentences, padding=True, truncation=True, return_tensors="pt")
        self.labels = torch.tensor(data_set.labels)
        self.dataset = TensorDataset(self.inputs.input_ids, self.inputs.attention_mask, self.labels)

        # DataLoaders for each task
        batch_size = 16
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)



class TestSet:
    def __init__(self, data_set, tokenizer):
        # Tokenizing test data for each task
        self.inputs = tokenizer(data_set.sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
        self.labels = torch.tensor(data_set.labels)



class Processor:
    def __init__(self, training_set):
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        self.task = TrainingSet(training_set, self.tokenizer)

        self.model = SingleTaskBERT(num_labels=self.task.size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-5)
        
    def handle_task(self, batch):
        input_ids, attention_mask, labels = [item.to('cpu') for item in batch]
        logits = self.model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()  # Accumulate gradients

    def train_model(self):
        self.model.train() # enter training

        for epoch in range(10):
            # zip with range(10) so the claim training doesn't take multiple hours
            for batch, _ in zip(self.task.dataloader, range(10)):
                self.optimizer.zero_grad()
                self.handle_task(batch)
                self.optimizer.step()

            print(f"Epoch {epoch+1} completed.")
            
        self.model.eval() # exit training

    def evaluate(self, test_set):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(test_set.inputs.input_ids, test_set.inputs.attention_mask)
            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            accuracy = (predictions == test_set.labels).float().mean()
        return accuracy.item()

    def accuracy_test(self, test_set):
        test = TestSet(test_set, self.tokenizer)
        accuracy = self.evaluate(test)
        print(f"Accuracy: {accuracy:.4f}")
    
    def model_save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def model_load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()



if __name__ == "__main__":
    data = Data()
    print("Training stance")
    processor = Processor(data.stance_train_set)
    processor.train_model()
    #processor.model_save("model_stance.pt")
    #processor.model_load("model_stance.pt")
    processor.accuracy_test(data.stance_dev_set)

    print("Training claim")
    processor = Processor(data.claim_train_set)
    processor.train_model()
    #processor.model_save("model_stance.pt")
    #processor.model_load("model_stance.pt")
    processor.accuracy_test(data.claim_dev_set)
