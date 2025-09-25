import pandas as pd
from collections import Counter
from tqdm import tqdm
import os
import shutil
import glob
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics import f1_score, accuracy_score


# Extract CROSSED/MIXED Keywords

VERSION = 1
FOLDER_NAME = 'KEY_FT_DISTIL_only_test'

df_id_train = pd.read_csv(f'/scratch/my-data/HS_DATA/HS_49/indo/SM-size/v{VERSION}/df_49_id-sm-v{VERSION}-train.csv', dtype={'hs_code':str})
df_id_valid = pd.read_csv(f'/scratch/my-data/HS_DATA/HS_49/indo/SM-size/v{VERSION}/df_49_id-sm-v{VERSION}-valid.csv', dtype={'hs_code':str})
df_id_tests = pd.read_csv(f'/scratch/my-data/HS_DATA/HS_49/indo/SM-size/v{VERSION}/df_49_id-sm-v{VERSION}-tests.csv', dtype={'hs_code':str})

len(df_id_train),len(df_id_valid),len(df_id_tests),

df_en_train = pd.read_csv(f'/scratch/my-data/HS_DATA/HS_49/india/SM-size/v{VERSION}/df_49_en-sm-v{VERSION}-train.csv', dtype={'hs_code':str})
df_en_valid = pd.read_csv(f'/scratch/my-data/HS_DATA/HS_49/india/SM-size/v{VERSION}/df_49_en-sm-v{VERSION}-valid.csv', dtype={'hs_code':str})
df_en_tests = pd.read_csv(f'/scratch/my-data/HS_DATA/HS_49/india/SM-size/v{VERSION}/df_49_en-sm-v{VERSION}-tests.csv', dtype={'hs_code':str})

len(df_en_train),len(df_en_valid),len(df_en_tests),


def create_df_most_words(most_common, lang, df_data):
    di_most_id = {}

    for seq, label in tqdm(enumerate(range(0,49))):
        
        li_words_id = []
        li_keywords = []
        
        for idx, row in df_data[df_data['label'] == label].iterrows():
            words = row['text'].split(' ')
            words = [word for word in words if len(word) > 3]
            li_words_id.extend(words)

        counter = Counter(li_words_id)
        di_most_id[seq] = [words[0] for words in counter.most_common(most_common)] 
    df_words = pd.DataFrame.from_records(di_most_id).transpose()
    
    li_cols = []
    
    for col in range(most_common):
        li_cols.append(f'{lang}_{col}')
        
    df_words.columns = li_cols
    
    li_keywords = df_words[li_cols].values.tolist()
    li_join_words = df_words[li_cols].apply(' '.join, axis=1).tolist()
    df_words['join'] = li_join_words
    df_words['keyword'] = li_keywords
    df_words['label'] = list(range(49))
    
    return df_words, li_join_words, li_keywords


def create_folders():
    root_path = f'/scratch/my-data/HS_DATA/HS_49/join_id_en/{FOLDER_NAME}'
    li_dir_1 = ['CROSSED'] # ['CROSSED', 'ALIGNED']
    li_dir_2 = ['KW-2']
    li_dir_3 = ['ds_key_few0','ds_key_few1','ds_key_few2','ds_key_few3']

    li_base_file = ['df_train_en_only_.csv',
                    'df_train_id_only_.csv',
                    'df_valid_en_only_.csv',            
                    'df_valid_id_only_.csv',
                    'df_tests_en_only_.csv',
                    'df_tests_id_only_.csv']

    # Create folders and subfolders
    for parent in li_dir_1:
        for child in li_dir_2:
            for sub in li_dir_3:
                # Construct the full path for each subfolder
                folder_path = os.path.join(root_path, parent, child, sub)
                # Create the folder (including parent directories if they don’t exist)
                os.makedirs(folder_path, exist_ok=True)
                print(f"Created: {folder_path}")


create_folders()

def folder_source(in_few):
    # li_few = ['few0','few1','few2','few3']
    li_files = []

    # Define the folder path and the search pattern
    folder_path = '/scratch/my-data/HS_DATA/HS_49/join_id_en/AUG/'
    file_pattern = f"**/*{in_few}*"  # Find all .txt files in all subfolders

    # Use glob to recursively find all .txt files
    file_paths = glob.glob(f"{folder_path}/{file_pattern}", recursive=True)

    for file_path in file_paths:
        li_files.append(file_path)
            
    return li_files


def folder_target(in_few):
    # li_few = ['few0','few1','few2','few3']
    li_files = []

    # Define the folder path and the search pattern
    folder_path = f'/scratch/my-data/HS_DATA/HS_49/join_id_en/{FOLDER_NAME}'
    file_pattern = f"**/*{in_few}*"  # Find all .txt files in all subfolders

    # Use glob to recursively find all .txt files
    file_paths = glob.glob(f"{folder_path}/{file_pattern}", recursive=True)

    # Print the list of matching files
    for file_path in file_paths:
        li_files.append(file_path)

    return sorted(li_files)


def copy_files():
    
    for few in ['few0','few1','few2','few3']:
        # Define source and destination folders
        source_folder = list(set(folder_source(few)))[0]
        destination_folders = list(set(folder_target(few)))
        
        for destination_folder in destination_folders:
            # List of specific filenames to copy
            files_to_copy = [
                             'df_train_id_only_.csv',
                             'df_train_en_only_.csv',
                             'df_valid_id_only_.csv',
                             'df_valid_en_only_.csv',
                             'df_tests_id_only_.csv',
                             'df_tests_en_only_.csv']

            # Ensure the destination folder exists
            os.makedirs(destination_folder, exist_ok=True)

            # Loop through the list and copy each specified file
            for filename in files_to_copy:
                source_file = os.path.join(source_folder, filename)
                destination_file = os.path.join(destination_folder, filename)

                # Check if the file exists before copying
                if os.path.exists(source_file):
                    shutil.copy(source_file, destination_file)
                    print(f"Copied: {source_file} to {destination_file}")
                else:
                    print(f"{filename} does not exist in the source folder.")


copy_files()

def create_keyword_files_MANUAL():
    most_commons = [1,2]
    stages = ['few0','few1','few2','few3']
    kw_types = ['ALIGNED','CROSSED']
    
    for most_common in most_commons:
        df_id_keywords, li_id_join_key, li_id_keywords = create_df_most_words(most_common, 'tr_id', df_id_train)
        df_en_keywords, li_en_join_key, li_en_keywords = create_df_most_words(most_common, 'tr_en', df_en_train)

        for kw_type in kw_types:
            for stage in stages:
                df_train_id_only_ = f'/scratch/my-data/HS_DATA/HS_49/join_id_en/{FOLDER_NAME}/{kw_type}/KW-{most_common}/ds_key_{stage}/df_train_id_only_.csv'
                df_train_en_only_ = f'/scratch/my-data/HS_DATA/HS_49/join_id_en/{FOLDER_NAME}/{kw_type}/KW-{most_common}/ds_key_{stage}/df_train_en_only_.csv'
                df_valid_id_only_ = f'/scratch/my-data/HS_DATA/HS_49/join_id_en/{FOLDER_NAME}/{kw_type}/KW-{most_common}/ds_key_{stage}/df_valid_id_only_.csv'
                df_valid_en_only_ = f'/scratch/my-data/HS_DATA/HS_49/join_id_en/{FOLDER_NAME}/{kw_type}/KW-{most_common}/ds_key_{stage}/df_valid_en_only_.csv'
                df_tests_id_only_ = f'/scratch/my-data/HS_DATA/HS_49/join_id_en/{FOLDER_NAME}/{kw_type}/KW-{most_common}/ds_key_{stage}/df_tests_id_only_.csv'
                df_tests_en_only_ = f'/scratch/my-data/HS_DATA/HS_49/join_id_en/{FOLDER_NAME}/{kw_type}/KW-{most_common}/ds_key_{stage}/df_tests_en_only_.csv'
                
                if kw_type == 'CROSSED':
                    # crossed
                    li_key_file = [(df_train_id_only_, df_en_keywords),
                                   (df_train_en_only_, df_id_keywords),
                                   (df_valid_id_only_, df_en_keywords),
                                   (df_valid_en_only_, df_id_keywords),           
                                   (df_tests_id_only_, df_en_keywords),
                                   (df_tests_en_only_, df_id_keywords),]
                else:
                    # aligned
                    li_key_file = [(df_valid_id_only_, df_id_keywords),
                                   (df_valid_en_only_, df_en_keywords),
                                   (df_train_id_only_, df_id_keywords),
                                   (df_train_en_only_, df_en_keywords),
                                   (df_tests_id_only_, df_id_keywords),
                                   (df_tests_en_only_, df_en_keywords),]

                for file_source in li_key_file:    
                    file, df_ref = file_source
                    df_data = pd.read_csv(file)
                    df_data.reset_index(inplace=True)
                    df_data.rename(columns={'index':'seq'}, inplace=True)
                    df_data = df_data.merge(df_ref, on='label', how='left')
                    df_data = df_data[['seq','keyword','label']]
                    df_data = df_data.sort_values('seq', ascending=True)
                    df_data.columns = ['seq','text','label']
                    destination_folder = os.path.dirname(file)
                    filename = f'{os.path.basename(file)[:11]}_keywo.csv'
                    print(os.path.join(destination_folder, filename))
                    df_data.to_csv(os.path.join(destination_folder, filename))
