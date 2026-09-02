import pandas as pd
import os
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
import json
import numpy as np
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


class ManifestFundusDataset(Dataset):
    def __init__(self, manifest_path, project_root, transform, partition='train', train_limit=0, val_limit=0):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        split_name = 'validation' if partition in ('val', 'validation') else partition
        records = self._resolve_split_records(manifest, split_name)
        self.ann = []
        for record in records:
            self.ann.extend(self._flatten_record(record))

        if split_name == 'train' and train_limit and train_limit > 0:
            self.ann = self.ann[:train_limit]
        if split_name == 'validation' and val_limit and val_limit > 0:
            self.ann = self.ann[:val_limit]

        self.project_root = project_root
        self.transform = transform

    def _resolve_split_records(self, manifest, split_name):
        if split_name in manifest:
            split_records = manifest[split_name]
        elif 'splits' in manifest and split_name in manifest['splits']:
            split_records = manifest['splits'][split_name]
        else:
            raise ValueError(f"Split '{split_name}' not found in manifest")
        if not isinstance(split_records, list):
            raise ValueError(f"Manifest split '{split_name}' must be a list")
        return split_records

    def _get_image_path(self, record):
        image_path = (
            record.get('image_path')
            or record.get('path')
            or record.get('image')
            or record.get('ImageID')
            or record.get('url')
        )
        if image_path is None:
            raise ValueError("Manifest record is missing an image path field")
        return image_path

    def _flatten_record(self, record):
        image_path = self._get_image_path(record)
        annotations = record.get('annotations')
        if not isinstance(annotations, list) or len(annotations) == 0:
            annotations = [record]

        flattened = []
        for ann in annotations:
            disease = ann.get('Disease', ann.get('disease', record.get('Disease', record.get('disease', ''))))
            description = ann.get('Description', ann.get('description', record.get('Description', record.get('description', ''))))
            flattened.append({
                'image_path': image_path,
                'disease': str(disease),
                'description': str(description),
            })
        return flattened

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        data_item = self.ann[index]
        image_path = data_item['image_path']
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.project_root, image_path)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        disease_text = f"This is a fundus image of {data_item['disease']}"
        description_text = data_item['description']
        return image, [disease_text], [description_text]


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