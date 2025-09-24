#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm
import pandas as pd
import os, time, datetime


# # KeywordEnhancedDistilBERT

# In[ ]:


class KeywordEnhancedDistilBERT(nn.Module):
    def __init__(self, num_labels=49, dropout_rate=0.2):
        super().__init__()
        
        self.text_encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        self.keyword_encoder = AutoModel.from_pretrained('distilbert-base-uncased')
        
        # Fusion 
        self.fusion = nn.Sequential(
            nn.Linear(768 * 2, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # classification layers
        self.classifier = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_labels)
        )
        
        self._init_weights()

    def _init_weights(self):
        for module in [self.fusion, self.classifier]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, input_ids, attention_mask, keyword_ids, keyword_mask):
        # main text
        text_outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # keywords
        keyword_outputs = self.keyword_encoder(keyword_ids, attention_mask=keyword_mask)
        keyword_outputs = keyword_outputs.last_hidden_state
        
        # Combine
        combined = torch.cat([text_features, keyword_outputs], dim=1)
        fused = self.fusion(combined)
        
        return self.classifier(fused)


# # TextKeywordDataset

# In[ ]:


class TextKeywordDataset(Dataset):
    def __init__(self, texts, keywords, labels, tokenizer, max_length=128, max_keywords_length=128):
        self.texts = texts
        self.keywords = keywords
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_keywords_length = max_keywords_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        keywords_text = ", ".join(self.keywords[idx])
        
        # tokenize text
        text_encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # tokenize keywords
        keyword_encoding = self.tokenizer(
            keywords_text,
            add_special_tokens=True,
            max_length=self.max_keywords_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': text_encoding['input_ids'].flatten(),
            'attention_mask': text_encoding['attention_mask'].flatten(),
            'keyword_ids': keyword_encoding['input_ids'].flatten(),
            'keyword_mask': keyword_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# In[ ]:


def evaluate_model(model, eval_loader, device, num_labels=49):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            keyword_ids = batch['keyword_ids'].to(device)
            keyword_mask = batch['keyword_mask'].to(device)
            
            outputs = model(input_ids, attention_mask, keyword_ids, keyword_mask)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].cpu().numpy())
    
    report = classification_report(all_labels, all_preds, labels=range(num_labels), output_dict=True)
    
    
    return report


# In[ ]:


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# In[ ]:


def train_model(model, train_loader, 
                       val_loader, 
                       tests_loader, 
                       tests_id_loader, 
                       tests_en_loader, 
                       num_epochs, learning_rate, device, model_path, num_warmup_steps=0):
    
    optimizer = AdamW([
        {'params': model.text_encoder.parameters(), 'lr': learning_rate},
        {'params': model.keyword_encoder.parameters(), 'lr': learning_rate},
        {'params': list(model.fusion.parameters()) + 
                   list(model.classifier.parameters()) + 
                   list(model.keyword_attention.parameters()), 
         'lr': learning_rate * 5}
    ], weight_decay=0.01)
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    best_val_loss = float('inf')
    best_model = None
    patience = 3
    patience_counter = 0
    train_losses=[]
    valid_losses=[]
    training_stats = []
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            keyword_ids = batch['keyword_ids'].to(device)
            keyword_mask = batch['keyword_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask, keyword_ids, keyword_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            train_steps += 1
            
            progress_bar.set_postfix({
                'training_loss': '{:.3f}'.format(total_train_loss / train_steps),
                'lr': '{:.1e}'.format(scheduler.get_last_lr()[0])
            })
        
        avg_train_loss = total_train_loss / train_steps
        train_losses.append(avg_train_loss)
        training_time = format_time(time.time() - start_time)
        
        # validation
        model.eval()
        total_val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                keyword_ids = batch['keyword_ids'].to(device)
                keyword_mask = batch['keyword_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, keyword_ids, keyword_mask)
                loss = criterion(outputs, labels)
                
                total_val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = total_val_loss / val_steps
        valid_losses.append(avg_val_loss)
        
        print(f'\nEpoch {epoch + 1}:')
        print(f'Average training loss: {avg_train_loss:.3f}')
        print(f'Average validation loss: {avg_val_loss:.3f}')
        
        training_stats.append(
            {
                'epoch_numb': epoch + 1,
                'train_loss': avg_train_loss,
                'valid_loss': avg_val_loss,
                'train_time': training_time
            }
        )
        
        df_stats = pd.DataFrame(data=training_stats)
        df_stats.to_csv(f'{model_path}/df_stat_{epoch}.csv')
        
        # testing
        report_all = evaluate_model(model, tests_loader, device)
        df_result = pd.DataFrame.from_dict(report_all)
        df_result = df_result.transpose()
        df_result.to_csv(f'{model_path}/df_resu_{epoch}.csv')
        
#         report_id = evaluate_model(model, tests_id_loader, device)
#         df_result_id = pd.DataFrame.from_dict(report_id)
#         df_result_id = df_result_id.transpose()
#         df_result_id.to_csv(f'{model_path}/df_r_id_{epoch}.csv')
        
#         report_en = evaluate_model(model, tests_en_loader, device)
#         df_result_en = pd.DataFrame.from_dict(report_en)
#         df_result_en = df_result_en.transpose()
#         df_result_en.to_csv(f'{model_path}/df_r_en_{epoch}.csv')
        
#     model_name = f'{model_path}/model_{epoch}.pt'
#     torch.save(model.state_dict(), model_name)
    
    return model


# In[ ]:


def run_model(train_texts, train_keywo, train_label, 
              valid_texts, valid_keywo, valid_label, 
              tests_texts, tests_keywo, tests_label, 
              tests_id_texts, tests_id_keywo, tests_id_label, 
              tests_en_texts, tests_en_keywo, tests_en_label, 
              model_path):
    
    # Hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 3
    LEARNING_RATE = 2e-5
    MAX_LENGTH = 128
    MAX_KEYWORDS_LENGTH = 128
    NUM_LABELS = 49

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Create datasets
    train_dataset = TextKeywordDataset(
        train_texts, 
        train_keywo, 
        train_label, 
        tokenizer,
        max_length=MAX_LENGTH, max_keywords_length=MAX_KEYWORDS_LENGTH
    )

    valid_dataset = TextKeywordDataset(
        valid_texts, 
        valid_keywo, 
        valid_label, 
        tokenizer,
        max_length=MAX_LENGTH, max_keywords_length=MAX_KEYWORDS_LENGTH
    )

    tests_dataset = TextKeywordDataset(
        tests_texts, 
        tests_keywo, 
        tests_label, 
        tokenizer,
        max_length=MAX_LENGTH, max_keywords_length=MAX_KEYWORDS_LENGTH
    )
    
    tests_id_dataset = TextKeywordDataset(
        tests_id_texts, 
        tests_id_keywo, 
        tests_id_label, 
        tokenizer,
        max_length=MAX_LENGTH, max_keywords_length=MAX_KEYWORDS_LENGTH
    )
    
    tests_en_dataset = TextKeywordDataset(
        tests_en_texts, 
        tests_en_keywo, 
        tests_en_label, 
        tokenizer,
        max_length=MAX_LENGTH, max_keywords_length=MAX_KEYWORDS_LENGTH
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    tests_loader = DataLoader(tests_dataset, batch_size=BATCH_SIZE)
    tests_id_loader = DataLoader(tests_id_dataset, batch_size=BATCH_SIZE)
    tests_en_loader = DataLoader(tests_en_dataset, batch_size=BATCH_SIZE)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = KeywordEnhancedDistilBERT(num_labels=NUM_LABELS).to(device)

    # Train model
    best_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        tests_loader=tests_loader,
        tests_id_loader=tests_id_loader,
        tests_en_loader=tests_en_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device,
        model_path=model_path,
        num_warmup_steps=100
    )
    
    return best_model


# In[100]:


# CV parameters
model_name = 'DISTIL'
cv_data_folder_EN = '/scratch/my-data/HS_DATA/HS_49/TF_IDF/CV_10_EN'
cv_data_folder_ID = '/scratch/my-data/HS_DATA/HS_49/TF_IDF/CV_10_ID'

# Run 10-fold cross validation
for fold in range(1, 11):  # folds 1 to 10
    print(f"\n{'='*50}")
    print(f"Starting Fold {fold}/10")
    print(f"{'='*50}")
    
    # Load training and test data for current fold
    train_file_EN = f'{cv_data_folder_EN}/df_train_fold_{fold:02d}.csv'
    train_file_ID = f'{cv_data_folder_ID}/df_train_fold_{fold:02d}.csv'
    test_file_EN = f'{cv_data_folder_EN}/df_test_fold_{fold:02d}.csv'
    test_file_ID = f'{cv_data_folder_ID}/df_test_fold_{fold:02d}.csv'
    
    # Load data
    df_train_en_only = pd.read_csv(train_file_EN)
    df_train_id_only = pd.read_csv(train_file_ID)
    df_tests_en_only = pd.read_csv(test_file_EN)
    df_tests_id_only = pd.read_csv(test_file_ID)
    
    # TRAINING
    train_texts = df_train_en_only['text'].tolist()    + df_train_id_only['text'].tolist()
    train_keywo = df_train_en_only['keyword'].tolist() + df_train_id_only['keyword'].tolist()
    train_label = df_train_en_only['label'].tolist()   + df_train_id_only['label'].tolist()

    assert(len(train_texts) == len(train_keywo) == len(train_label))

    # TESTS
    tests_texts = df_tests_en_only['text'].tolist()    + df_tests_id_only['text'].tolist()
    tests_keywo = df_tests_en_only['keyword'].tolist() + df_tests_id_only['keyword'].tolist()
    tests_label = df_tests_en_only['label'].tolist()   + df_tests_id_only['label'].tolist()

    assert(len(tests_texts) == len(tests_keywo) == len(tests_label))
    
    print(f"Fold {fold}")
    
    ########################## small datasets ############################

#     small_train = 500
#     small_tests = 100

#     train_texts = train_texts[:small_train]
#     train_keywo = train_keywo[:small_train]
#     train_label = train_label[:small_train]
#     tests_texts = tests_texts[:small_tests]
#     tests_keywo = tests_keywo[:small_tests]
#     tests_label = tests_label[:small_tests]
    
    ######################################################################
    
    # Create model path for current fold
    model_path = f'/scratch/my-model-scr/multilanguage/model_finetune_sbert/KEYWORD_CV/{model_name}/FOLD_{fold:02d}/'
    os.makedirs(model_path, exist_ok=False)

    # Run training for current fold    
    best_model = run_model(train_texts, train_keywo, train_label, 
                           tests_texts, tests_keywo, tests_label, 
                           tests_texts, tests_keywo, tests_label, 
                           tests_texts, tests_keywo, tests_label, 
                           tests_texts, tests_keywo, tests_label, 
                           model_path)


    model_save = f'{model_path}/model.pt'
    torch.save(best_model.state_dict(), model_save)
#     break


# In[ ]:




