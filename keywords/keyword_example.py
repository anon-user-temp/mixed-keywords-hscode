#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datasets
import transformers
from transformers import Trainer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import DataCollatorWithPadding
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


# # load dataset

# In[2]:


def join_ds(join_ds, df_id_train, df_id_valid, df_id_tests, df_en_train, df_en_valid, df_en_tests):
    
    df_id_train = df_id_train[['text','label']]
    df_id_valid = df_id_valid[['text','label']]
    df_id_tests = df_id_tests[['text','label']]
    
    df_en_train = df_en_train[['text','label']]
    df_en_valid = df_en_valid[['text','label']]
    df_en_tests = df_en_tests[['text','label']]
    
    if (join_ds == 'id-en'):
        df_train = pd.concat([df_id_train, df_en_train])
        df_train.reset_index(inplace=True, drop=True)
        
        df_valid = pd.concat([df_id_valid, df_en_valid])
        df_valid.reset_index(inplace=True, drop=True)

        df_tests = pd.concat([df_id_tests, df_en_tests])
        df_tests.reset_index(inplace=True, drop=True)
    
    elif (join_ds == 'id'):
        df_train = df_id_train
        df_valid = df_id_valid
        df_tests = df_id_tests
    
    elif (join_ds == 'en'):
        df_train = df_en_train
        df_valid = df_en_valid
        df_tests = df_en_tests
        
    return df_train, df_valid, df_tests


# In[3]:


def create_ds(df_train, df_valid, df_tests):
    ds_train = datasets.Dataset.from_pandas(df_train[['text','label']])
    ds_valid = datasets.Dataset.from_pandas(df_valid[['text','label']])
    ds_tests = datasets.Dataset.from_pandas(df_tests[['text','label']])
    
    ds_dict = datasets.DatasetDict({'train':ds_train, 
                                    'valid':ds_valid, 
                                    'tests':ds_tests})
    
    return ds_dict


# In[4]:


def tokenised_ds(model_encoder, ds_dict):
    
    def tokenize_text(example):
        return tokenizer(example['text'], 
                     truncation=True, 
                     max_length=128, 
                     padding='max_length')
    
    tokenizer = AutoTokenizer.from_pretrained(model_encoder, clean_up_tokenization_spaces=True) 
    tokenized_ds = ds_dict.map(tokenize_text, batched=True)
    
    return tokenized_ds


# In[5]:


def full_train(tokenized_ds, model_encoder, num_labels, model_path):
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        
        return {'accuracy':acc, 'f1':f1}
    
    tokenizer = AutoTokenizer.from_pretrained(model_encoder, clean_up_tokenization_spaces=True) 
    model = AutoModelForSequenceClassification.from_pretrained(model_encoder, num_labels=num_labels)
    
    logging_steps = tokenized_ds['train'].num_rows/16
    
    training_args = TrainingArguments(output_dir=model_path, 
                                      num_train_epochs=3,
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=16,
                                      evaluation_strategy='epoch',
                                      logging_strategy='epoch',
                                      learning_rate=2e-05,
                                      logging_steps=50,
                                      save_strategy='epoch',
                                      overwrite_output_dir=True,
                                      load_best_model_at_end=True,
                                      save_total_limit=1,
                                )
    
    trainer = Trainer(
              model,
              training_args,
              train_dataset=tokenized_ds["train"],
              eval_dataset=tokenized_ds["valid"],
              tokenizer=tokenizer,
              compute_metrics=compute_metrics
    )

    trainer.train()
    
    return trainer 


# In[6]:


def train_all_model(df_model, model_path, number_labels, lang,
                    df_id_train, df_id_valid, df_id_tests,
                    df_en_train, df_en_valid, df_en_tests):
    
    for idx, row in tqdm(df_model.iterrows()):
        path_name = row['path_name']
        encoder = row['encoder']
        
        for lang in [lang]:
            df_train, df_valid, df_tests = join_ds(lang, 
                                                   df_id_train, df_id_valid, df_id_tests,
                                                   df_en_train, df_en_valid, df_en_tests)
            
            ds_dict = create_ds(df_train, df_valid, df_tests)
            
            model_full_path = f'{model_path}/{path_name}_{lang}'

            tokenized_ds = tokenised_ds(encoder, ds_dict)
            trainer = full_train(tokenized_ds, encoder, number_labels, model_full_path)

            preds_output = trainer.predict(tokenized_ds['tests'])
            y_pred = np.argmax(preds_output.predictions, axis=1)
            y_true = preds_output.label_ids
            df_result = pd.DataFrame(classification_report(y_true, y_pred, digits=4, output_dict=True)).transpose()
            df_result.reset_index(inplace=True)
            
            df_result.rename(columns={'index':'label'}, inplace=True)
            df_result.to_csv(f'{model_full_path}/{path_name}_{lang}_result.csv', index=False)

#     return tokenized_ds


# # parameters

# In[16]:


kw_num = '1'
stage = f'KW-{kw_num}-JOIN'
lang = 'id-en'

di_baseline_models = {
    'path_name':[f'FT_C40_T128_E3_DistilKeyword_{stage}'],
    'encoder':['distilbert-base-uncased']
}

df_model = pd.DataFrame(di_baseline_models)
model_path = '/scratch/my-model-scr/multilanguage/model_baseline/ft-keyword-join'

# KEYWORD for
df_id_train = pd.read_csv(f'/scratch/my-data/HS_DATA/HS_49/join_id_en/DATASET_KEYWORD_FINE_TUNE/{stage}/df_40_tr-id-sm-v5-train.csv', dtype={'hs_code':str})
df_id_valid = pd.read_csv(f'/scratch/my-data/HS_DATA/HS_49/join_id_en/DATASET_KEYWORD_FINE_TUNE/{stage}/df_40_tr-id-sm-v5-valid.csv', dtype={'hs_code':str})
df_id_tests = pd.read_csv(f'/scratch/my-data/HS_DATA/HS_49/join_id_en/DATASET_KEYWORD_FINE_TUNE/{stage}/df_40_tr-id-sm-v5-tests.csv', dtype={'hs_code':str})
df_en_train = pd.read_csv(f'/scratch/my-data/HS_DATA/HS_49/join_id_en/DATASET_KEYWORD_FINE_TUNE/{stage}/df_43_tr-en-sm-v5-train.csv', dtype={'hs_code':str})
df_en_valid = pd.read_csv(f'/scratch/my-data/HS_DATA/HS_49/join_id_en/DATASET_KEYWORD_FINE_TUNE/{stage}/df_43_tr-en-sm-v5-valid.csv', dtype={'hs_code':str})
df_en_tests = pd.read_csv(f'/scratch/my-data/HS_DATA/HS_49/join_id_en/DATASET_KEYWORD_FINE_TUNE/{stage}/df_43_tr-en-sm-v5-tests.csv', dtype={'hs_code':str})


# In[17]:


result = train_all_model(df_model, model_path, number_labels, lang, 
                         df_id_train, df_id_valid, df_id_tests, 
                         df_en_train, df_en_valid, df_en_tests)


