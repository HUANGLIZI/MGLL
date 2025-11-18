import pandas as pd
import os
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
import json
import numpy as np
from models import Tokenizer
import torch
from torch.utils.data import Dataset


class PretrainMMFundusDataset(Dataset):
    def __init__(self, data_dir, json_list, transform, max_words=512, partition='train', tokenizer_path=None):

        self.json_list = json_list
        if partition == 'train':
            ann = []
            fileHandler = open(os.path.join(data_dir, json_list + '_train.txt'), 'r')
            listOfLines  =  fileHandler.readlines()
            for line in listOfLines:
                json_path = data_dir + line.strip()
                ann += json.load(open(json_path))
            self.ann = ann
            self.data_dict = {}
            
        else:
            ann = []
            fileHandler = open(os.path.join(data_dir, json_list + '_val.txt'), 'r')
            listOfLines  =  fileHandler.readlines()
            for  line in  listOfLines:
                json_path = data_dir + line.strip()
                ann += json.load(open(json_path))
            self.ann = ann
            self.data_dict = {}

        self.data_dir = data_dir
        self.transform = transform
        self.max_words = max_words
        self.max_keywords = 32
        tokenizer = Tokenizer(model_path=tokenizer_path)
        self.tokenizer1 = tokenizer
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        Keyword_list = []
        Desc_list = []
        Modality_list = []
        data_item = self.ann[index]
        
        if 'ImageID' in data_item.keys():
            url = data_item['ImageID']
            if self.json_list == "original_list":
                Keyword = "This is a fundus image of " + data_item['Keyword']
                Keyword_list.append(Keyword)
            elif self.json_list == "2level_list":
                disease = "This is a fundus image of " + data_item['Disease']
                description = data_item['Description']
                Keyword_list.append(disease)
                Desc_list.append(description)
            elif self.json_list == "2level_multi_disease_list":
                disease = "This is a fundus image of " + data_item['Disease']
                description = data_item['Description']
                Keyword_list.append(disease)
                Desc_list.append(description)
            elif self.json_list == "MIDRC_list":
                Modality = "This is a " + data_item['modality'] + " image."
                disease =  data_item['study_description_0']
                description = data_item['series_description']
                Modality_list.append(Modality)
                Keyword_list.append(disease)
                Desc_list.append(description)
            elif self.json_list == "only_FIVES_list":
                Keyword = "This is a fundus image of " + data_item['Keyword']
                Keyword_list.append(Keyword)
            elif self.json_list == "flair_list":
                Keyword = "This is a fundus image of " + data_item['Keyword']
                Keyword_list.append(Keyword)
            else:
                raise ValueError("No json_list specified")
            
            if self.json_list == "MIDRC_list":
                filename = url
            else:
                filename = self.data_dir + '/All_data_npz' + os.path.splitext(url[9:])[0] + '.npz'
            # image = Image.open(filename).convert('RGB')
            try:
                image = np.load(filename)['image']
                image = Image.fromarray(image)
            except:
                print(f"Error loading {filename}")
                return None, None, None

            image = self.transform(image)
                     
        else:
            raise ValueError("No image_id in data_item")
        if self.json_list == "MIDRC_list":
            return image, Keyword_list, Desc_list, Modality_list
        else:
            return image, Keyword_list, Desc_list


class FinetuneMMFundusDataset(Dataset):
    def __init__(self, data_dir, csv_name, transform, partition='train'):

        self.csv_name = csv_name
        
        if partition == 'train':
            csv_path = os.path.join(data_dir, 'All_csv_downstream', csv_name + '_Train.csv')
        elif partition == 'val':
            csv_path = os.path.join(data_dir, 'All_csv_downstream', csv_name + '_Val.csv')
        else:
            csv_path = os.path.join(data_dir, 'All_csv_downstream', csv_name + '_Test.csv')
        self.dataframe = pd.read_csv(csv_path)
        if "MIDRC" in self.csv_name:
            if "XR_Portable" in self.csv_name:
                transformer_file = pd.read_excel('/Datasets/MIDRC/label.xlsx', sheet_name='XR_portable')
            else:
                transformer_file = pd.read_excel('/Datasets/MIDRC/label.xlsx', sheet_name='XR')
            # keep only the "ImageID" and "loinc_long_common_name_0" columns in self.dataframe
            self.dataframe = self.dataframe[['ImageID', 'loinc_long_common_name_0']]
            # transform the "loinc_long_common_name_0" column in self.dataframe as the transformer_file
            self.dataframe['loinc_long_common_name_0'] = self.dataframe['loinc_long_common_name_0'].apply(lambda x: transformer_file[transformer_file['Lonic Long Common Name'] == x]['Label'].values[0])
        else:
            self.dataframe.iloc[:, 1:] = self.dataframe.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        self.data_dir = data_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        data_item = self.dataframe.iloc[index]
        url = data_item.iloc[0]
        labels = data_item.iloc[1:].apply(pd.to_numeric, errors='coerce').values
        labels = torch.tensor(labels, dtype=torch.float32)
        if "MIDRC" in self.csv_name:
            filename = url
        else:
            filename = self.data_dir + '/All_data_npz' + os.path.splitext(url[9:])[0] + '.npz'
        name = url.split('/')[-1]
        # image = Image.open(filename).convert('RGB')
        image = np.load(filename)['image']
        image = Image.fromarray(image)
        
        image = self.transform(image)
        
        return image, labels, name