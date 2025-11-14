from datasets import load_dataset, load_from_disk
import os
import numpy as np
from tqdm import tqdm
import pandas as pd


import Feature_extraction_functions as Fef



class Custom_DataSet_Manager():
    
    #Checks if there is dataset folder present, if not it creates it
    def __init__(self, DataSet_path, train_split, val_split, test_split, random_state):
        self.dataset_path = DataSet_path
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.random_state = random_state
        
        self.flag_file = os.path.join(self.dataset_path, "download_complete.flag")
        
        
        
    def download_database(self, dataset_name):
        
        if not self.is_downloaded():
            #Create folder if not present 
            os.makedirs(self.dataset_path, exist_ok=True)
    
            print("Downloading dataset...")
            dataset = load_dataset(dataset_name)
            dataset.save_to_disk(self.dataset_path)
            
            #Add flag but only if the dataset is completed. 
            #If downloading above is interrupted then its not present
            with open(self.flag_file, "w") as f:
                f.write("downloaded")
    
            print("Dataset downloaded and flagged!")
        
        else:
            print("Dataset is alredy downloaded!")
        
    def is_downloaded(self):
        # Check if the flag file exists
        return os.path.exists(self.flag_file)     
    
    def load_dataset_from_disk(self):
        #Check for flag
        if not self.is_downloaded():
            raise RuntimeError("Dataset not downloaded or incomplete. Download it first")
            
        #Load it to split it on run
        Dataset = load_from_disk(self.dataset_path)
        
        train, val, test = self.split_dataset(Dataset)
        return train, val, test
    
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
        
    
    def transform_to_features(self):
        # Paths for saved feature files
        df_train_file = os.path.join(self.dataset_path, "features_train.pkl")
        df_val_file   = os.path.join(self.dataset_path, "features_val.pkl")
        df_test_file  = os.path.join(self.dataset_path, "features_test.pkl")
        flag_file     = os.path.join(self.dataset_path, "features_extracted.flag")
        
        # Check if features were already extracted
        if os.path.exists(flag_file):
            print("Features already extracted, loading from disk...")
            df_train = pd.read_pickle(df_train_file)
            df_val   = pd.read_pickle(df_val_file)
            df_test  = pd.read_pickle(df_test_file)
            return df_train, df_val, df_test
    
        # Otherwise, extract features as before
        train, val, test = self.load_dataset_from_disk()
    
        def extract_features(dataset):
            rows = []
            c = 0
            for img_obj in tqdm(dataset):
                c+=1
                if c >1000:
                    break
    
                feature_dict = {}
                feature_dict['label'] = img_obj['style']
                
                img = np.array(img_obj['image'], dtype=np.uint8)
    
                # Feature extraction
                ########################################
                #1
                vert_sym, horiz_sym = Fef.symmetry_scores_ssim(img) 
                feature_dict['vert_sym'] = vert_sym
                feature_dict['horiz_sym'] = horiz_sym
                
                #2
                entropy = Fef.img_entropy(img, n_bins=64)
                feature_dict['entropy'] = entropy
                
                #3
                #f_spectrum = Fef.radial_power_spectrum(img, n_bins=64)
                #feature_dict['f_spectrum'] = f_spectrum
                
                #4
                color_moments_dict = Fef.color_moments(img)
                feature_dict.update(color_moments_dict)
                
                #5
                Colorfullness_dict = Fef.colorfulness_saturation(img)
                feature_dict.update(Colorfullness_dict)
                
                #6
                color_harmony_dict = Fef.color_harmony_contrast(img)
                feature_dict.update(color_harmony_dict)
                
                #7
                #lbp_histogram = Fef.lbp_histogram(img)
                #feature_dict['lbp_histogram'] = lbp_histogram
                
                #9
                c_stats_dict = Fef.contour_statistics(img)
                feature_dict.update(c_stats_dict)
                
                # 10. Haralick texture features
                haralick_dict = Fef.haralick_features(img)
                feature_dict.update(haralick_dict)
                
                # 11. Gabor filter bank features
                gabor_dict = Fef.gabor_features(img)
                feature_dict.update(gabor_dict)
                
                # 12. HOG features
                #hog_feat = Fef.hog_features(img)
                #feature_dict['hog_features'] = hog_feat
                
                # 13. Radial power spectrum / spectral slope
                spectral_dict = Fef.radial_spectral_summary(img)
                feature_dict.update(spectral_dict)
                
                # 14. Fractal dimension
                fd = Fef.fractal_dimension(img)
                feature_dict['fractal_dim'] = fd
                
                
                
                rows.append(feature_dict)
            
            df = pd.DataFrame(rows)
            return df
    
        df_train = extract_features(train)
        df_val   = extract_features(val)
        df_test  = extract_features(test)
        
        # Save to disk
        df_train.to_pickle(df_train_file)
        df_val.to_pickle(df_val_file)
        df_test.to_pickle(df_test_file)
        
        # Create flag file
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












