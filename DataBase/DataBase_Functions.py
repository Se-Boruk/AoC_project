from datasets import load_dataset, load_from_disk
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

import kagglehub
from PIL import Image
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor 

import Feature_extraction_functions as Fef


def extract_single_image_features(row_tuple):
    """Worker function to process one image on one CPU core."""
    _, row = row_tuple
    try:
        with Image.open(row['path']) as img_raw:
            #RGB mode
            img_rgb = img_raw.convert('RGB')
            
            #Resizing while keeping Aspect Ratio
            #.thumbnail() resizes the image in-place to fit inside the 512x512 box.
            img_rgb.thumbnail((512, 512), Image.Resampling.BILINEAR)
            
            img = np.array(img_rgb, dtype=np.uint8)

        feature_dict = {'label': row['style']}
        

        #Feature extraction
        ##################################
        v_sym, h_sym = Fef.symmetry_scores_ssim(img) 
        feature_dict.update({'vert_sym': v_sym, 'horiz_sym': h_sym})
        feature_dict['entropy'] = Fef.img_entropy(img, n_bins=64)
        feature_dict.update(Fef.color_moments(img))
        feature_dict.update(Fef.colorfulness_saturation(img))
        feature_dict.update(Fef.color_harmony_contrast(img))
        feature_dict.update(Fef.contour_statistics(img))
        feature_dict.update(Fef.haralick_features(img))
        feature_dict.update(Fef.gabor_features(img))
        feature_dict.update(Fef.radial_spectral_summary(img))
        feature_dict['fractal_dim'] = Fef.fractal_dimension(img)
        ##################################
        
        return feature_dict
    
    except Exception:
        return None



class Custom_DataSet_Manager():
    
    #Checks if there is dataset folder present, if not it creates it
    def __init__(self, DataSet_path, train_split, val_split, test_split, random_state):
        self.dataset_path = DataSet_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        self.flag_file = os.path.join(self.dataset_path, "download_complete.flag")
        
        
        
    def download_database(self, dataset_name="steubk/wikiart"):
        # Force kagglehub to use your specific folder as the cache
        os.environ["KAGGLEHUB_CACHE"] = self.dataset_path 
        
        if not self.is_downloaded():
            print(f"Downloading {dataset_name} to {self.dataset_path}...")
            # This will now download directly into your specified folder
            path = kagglehub.dataset_download(dataset_name)
            
            with open(self.flag_file, "w") as f:
                f.write(path)
            print("Download complete!")
        else:
            print("Dataset already present.")
        
        
    def is_downloaded(self):
        #Check if the flag file exists
        return os.path.exists(self.flag_file)     
    
    def load_dataset_from_disk(self, full_dataset_path):
        if not self.is_downloaded():
            raise RuntimeError("Dataset not downloaded. Run download_database first.")
            

        
        #Gather all file paths and labels
        data = []
        for style in os.listdir(full_dataset_path):
            style_path = os.path.join(full_dataset_path, style)
            
            if os.path.isdir(style_path):
                for img_name in os.listdir(style_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        data.append({'path': os.path.join(style_path, img_name), 'style': style})
        
        df = pd.DataFrame(data)
    
        #Split with given ratio
        train_df, temp_df = train_test_split(
            df, train_size=self.train_split, random_state=self.random_state, stratify=df['style']
        )
        
        val_ratio = self.val_split / (self.val_split + self.test_split)
        val_df, test_df = train_test_split(
            temp_df, train_size=val_ratio, random_state=self.random_state, stratify=temp_df['style']
        )
        
        return train_df, val_df, test_df
    
    def split_dataset(self,dataset):
        #Split dataset into train, val and test. Ready for work :). 
        #OFc with given random state or diseaster 
        
        #Train bc this dataset (at least the one for reconstruction - upscaling one may 
        #be different and require changes/addons) has only train. 
        #We need to split it on our own
        
        #Just load the data and shuffle it (so we mix the classes and hopefully mix them uniformly for training)
        #Cant use stratifying as we do not know the classes a priori (unsupervised learning)
        Data =  dataset["train"].shuffle(seed=self.random_state)
        
        #Split it into train and subset
        split_dataset = Data.train_test_split(test_size= (1 -self.train_split) , seed=self.random_state)
        
        train_subset = split_dataset['train']
        subset = split_dataset['test']
        
        #Split the subset into the val and test 
        test_fraction = self.val_split / ((self.val_split + self.test_split))
        
        split_dataset_1 = subset.train_test_split(test_size= test_fraction , seed=self.random_state)
        
        val_subset = split_dataset_1['train']
        test_subset = split_dataset_1['test']

        return train_subset, val_subset, test_subset
        
    
    def transform_to_features(self, full_dataset_path):
            df_train_file = os.path.join(self.dataset_path, "features_train.pkl")
            df_val_file   = os.path.join(self.dataset_path, "features_val.pkl")
            df_test_file  = os.path.join(self.dataset_path, "features_test.pkl")
            flag_file     = os.path.join(self.dataset_path, "features_extracted.flag")
            
            if os.path.exists(flag_file):
                print("Features already extracted, loading from disk...")
                return pd.read_pickle(df_train_file), pd.read_pickle(df_val_file), pd.read_pickle(df_test_file)
            
            train, val, test = self.load_dataset_from_disk(full_dataset_path)
            
            def run_parallel_extraction(df, desc):
                # Check available cores
                cores = os.cpu_count()
                print(f"Total CPU cores available: {cores}")
                #cores = max(1, cores - 1)
                print(f"Using {cores} cores")
                
                with ProcessPoolExecutor(max_workers=cores) as executor:
                    #Chunks of images per thread (batches)
                    results = list(tqdm(
                        executor.map(extract_single_image_features, df.iterrows(), chunksize=64), 
                        total=len(df), 
                        desc=desc
                    ))
                return pd.DataFrame([r for r in results if r is not None])
    
            print("Starting parallel feature extraction...")
            df_train = run_parallel_extraction(train, "Extracting Train")
            df_val   = run_parallel_extraction(val, "Extracting Val")
            df_test  = run_parallel_extraction(test, "Extracting Test")
            
            df_train.to_pickle(df_train_file)
            df_val.to_pickle(df_val_file)
            df_test.to_pickle(df_test_file)
            
            with open(flag_file, "w") as f:
                f.write("features extracted")
            
            return df_train, df_val, df_test
            
            
            
class LabelEncoderDF:
    def __init__(self):
        self.label_to_int = {}
        self.int_to_label = {}
        self.next_int = 0

    def encode_label(self, label):
        if label not in self.label_to_int:
            self.label_to_int[label] = self.next_int
            self.int_to_label[self.next_int] = label
            self.next_int += 1
        return self.label_to_int[label]

    def transform_df(self, df, label_col="label"):
        """
        Replace string labels in a DataFrame with integers.
        Returns the same DataFrame with label column overwritten.
        """
        df[label_col] = df[label_col].apply(self.encode_label)
        return df            
        

def Prepare_data_from_features(df, label_col="label"):
    """
    Flatten all features in a DataFrame into a single 2D array for ML,
    automatically detecting vector-like columns and flattening them.
    
    Args:
        df (pd.DataFrame): DataFrame with features and label
        label_col (str): Column name for the label
    
    Returns:
        X (np.array): 2D array of shape (n_samples, n_features)
        y (np.array): 1D array of integer labels
    """
    X_list = []

    for _, row in df.iterrows():
        features = []
        for col in df.columns:
            if col == label_col:
                continue
            val = row[col]
            # If value is iterable but not a string, treat as vector
            if isinstance(val, (list, tuple, np.ndarray)):
                features.extend(val)
            else:
                features.append(val)
        X_list.append(features)

    X = np.array(X_list, dtype=np.float32)
    y = df[label_col].values.astype(np.int64)

    return X, y












