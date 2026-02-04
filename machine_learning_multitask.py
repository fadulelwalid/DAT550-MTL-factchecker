# modified from
# https://github.com/uis-no/dat550-2024/blob/main/handson_multitask_learning/multitask_train.py

import torch
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import sys

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



class MultiTaskBERT(nn.Module):
    def __init__(self, num_labels_task1, num_labels_task2):
        """
        Initialization of the multitask model.

        Parameters:
        - num_labels_task1: number of unique labels for task 1
        - num_labels_task2: number of unique labels for task 2

        Returns:
        - MultiTaskBERT: the multitask neural network with the bert encoder.
        """
        super(MultiTaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Task-specific layers
        self.dropout = nn.Dropout(0.1)
        self.classifier_task1 = nn.Linear(self.bert.config.hidden_size, num_labels_task1)
        self.classifier_task2 = nn.Linear(self.bert.config.hidden_size, num_labels_task2)
        
    def forward(self, input_ids, attention_mask, task):
        """
        Forward pass for multitask learning.
        
        Parameters:
        - input_ids: Tensor of input IDs
        - attention_mask: Tensor for attention mask
        - task: Integer specifying the task (1 for task1, 2 for task2)
        
        Returns:
        - logits: Task-specific logits
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        pooled_output = self.dropout(outputs.pooler_output) # zero out 10% of the output at random
        
        # Determine which task is being asked for and use the appropriate classifier
        if task == 1:
            logits = self.classifier_task1(pooled_output)
        elif task == 2:
            logits = self.classifier_task2(pooled_output)
        else:
            raise ValueError("Invalid task identifier.")
        
        return logits



class TrainingSet:
    def __init__(self, data_set, tokenizer):
        self.size = len(data_set.labels_to_text)

        # Tokenize and prepare datasets separately for each task
        self.inputs = tokenizer(data_set.sentences, padding=True, truncation=True, return_tensors="pt")
        self.labels = torch.tensor(data_set.labels)
        self.dataset = TensorDataset(self.inputs.input_ids, self.inputs.attention_mask, self.labels)

        # DataLoaders for each task
        batch_size = 2
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)



class TestSet:
    def __init__(self, data_set, tokenizer):
        # Tokenizing test data for each task
        self.inputs = tokenizer(data_set.sentences, return_tensors='pt', padding=True, truncation=True, max_length=512)
        self.labels = torch.tensor(data_set.labels)



class Processor:
    def __init__(self):
        self.data = Data()

        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        self.task1 = TrainingSet(self.data.claim_train_set, self.tokenizer)
        self.task2 = TrainingSet(self.data.stance_train_set, self.tokenizer)

        self.model = MultiTaskBERT(num_labels_task1=self.task1.size, num_labels_task2=self.task2.size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-5)
        
    def handle_task(self, batch, task_num):
        input_ids, attention_mask, labels = [item.to("cpu") for item in batch]
        logits = self.model(input_ids, attention_mask, task=task_num)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()  # Accumulate gradients

    def train_model(self):
        self.model.train() # enter training

        for epoch in range(10):
            for (batch1, batch2) in zip(self.task1.dataloader, self.task2.dataloader):
                self.optimizer.zero_grad()
                self.handle_task(batch1, 1)
                self.handle_task(batch2, 2)
                self.optimizer.step()

            print(f"Epoch {epoch+1} completed.")
            
        self.model.eval() # exit training

    def evaluate(self, test_set, task_num):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(test_set.inputs.input_ids, test_set.inputs.attention_mask, task=task_num)
            predictions = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            accuracy = (predictions == test_set.labels).float().mean()
        return accuracy.item()

    def accuracy_test(self):
        test_task1 = TestSet(self.data.claim_dev_set, self.tokenizer)
        test_task2 = TestSet(self.data.stance_dev_set, self.tokenizer)

        # Evaluate Task 1 (Claim Detection)
        accuracy_task1 = self.evaluate(test_task1, 1)
        print(f"Task 1 (Claim Detection) Accuracy: {accuracy_task1:.4f}")

        # Evaluate Task 2 (Stance Detection)
        accuracy_task2 = self.evaluate(test_task2, 2)
        print(f"Task 2 (Stance Detection) Accuracy: {accuracy_task2:.4f}")
    
    def model_save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def model_load(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
    
    def take_user_input(self):
        while True:
            text = input("write a claim: ")
            inputs = self.tokenizer([text], return_tensors='pt', padding=True, truncation=True, max_length=512)
            output = self.model(inputs.input_ids, inputs.attention_mask, task=1)
            claim_result = self.data.claim_train_set.get_label_text(int(torch.argmax(torch.softmax(output, dim=1), dim=1)[0]))
            output = self.model(inputs.input_ids, inputs.attention_mask, task=2)
            stance_result = self.data.stance_train_set.get_label_text(int(torch.argmax(torch.softmax(output, dim=1), dim=1)[0]))

            print(f"claim detector label: {claim_result}")
            print(f"stance detector label {stance_result}")

            is_claim = False

            if claim_result == "No":
                is_claim = False
            if claim_result == "Yes":
                is_claim = True

            is_fact = False

            if stance_result == "REFUTES":
                is_fact = False
            if stance_result == "SUPPORTS":
                is_fact = True
            if stance_result == "NOT ENOUGH INFO":
                is_fact = None
            
            if not is_claim:
                print("The statement doesn't claim anything")
            else:
                if is_fact is True:
                    print("The statement is True")
                elif is_fact is False:
                    print("The statement is False")
                else:
                    print("I don't know")
            

            


if __name__ == "__main__":

    args = dict()
    args["train"] = False
    args["save"] = False
    args["load"] = False
    args["accuracy"] = False
    args["interactive"] = False

    for arg in sys.argv:
        args[arg] = True

    processor = Processor()

    if args["train"]:
        processor.train_model()
    
    if args["save"]:
        processor.model_save("model.pt")

    if args["load"]:
        processor.model_load("model.pt")
    
    if args["accuracy"]:
        processor.accuracy_test()
    
    if args["interactive"]:
        processor.take_user_input()