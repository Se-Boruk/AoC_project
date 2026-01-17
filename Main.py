###################################################################
# ( 0 ) Libs and dependencies 
###################################################################

#Basic libs
import sys
import os
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import pickle

from Config import DATABASE_FOLDER, DATASET_PATH, DATASET_NAME, FULL_DATASET_PATH, PROCESSED_DATA_PATH
from Config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_STATE



##########################################################################################
#Move folder up to go to database folder to use manager from here
sys.path.insert(0, DATABASE_FOLDER)
from DataBase_Functions import Custom_DataSet_Manager, LabelEncoderDF, Prepare_data_from_features
import Functions

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


if __name__ == '__main__':
    ###################################################################
    # ( 1 ) Loading data
    ###################################################################
    if os.path.exists(PROCESSED_DATA_PATH):
        print("Loading already processed data from disk...")
        with open(PROCESSED_DATA_PATH, 'rb') as f:
            Processed_data = pickle.load(f)
    
    else:
        
        #Data loading (Not stored in RAM but is accessed on demand)
        
        #Load manager and execute
        manager = Custom_DataSet_Manager(DataSet_path = DATASET_PATH,
                                         train_split = TRAIN_SPLIT,
                                         val_split = VAL_SPLIT,
                                         test_split = TEST_SPLIT,
                                         random_state = RANDOM_STATE
                                         )
        
        #Download data if not present
        manager.download_database(DATASET_NAME)
        #Load dataset
        Train_set, Val_set, Test_set = manager.load_dataset_from_disk(full_dataset_path = FULL_DATASET_PATH)
        
        
        
        ###################################################################
        # ( 2 ) Extracting features from image
        ###################################################################
        Train_features, Val_features, Test_features = manager.transform_to_features(full_dataset_path = FULL_DATASET_PATH)
        
        
        #Drop nans values
        print("Dropping NaNs...")
        Train_features = Train_features.dropna().reset_index(drop=True)
        Val_features   =  Val_features.dropna().reset_index(drop=True)
        Test_features  = Test_features.dropna().reset_index(drop=True)
        
        print("Labeling...")
        #replacing labels with numbers
        Label_encoder = LabelEncoderDF()
        Train_features = Label_encoder.transform_df(Train_features)
        Val_features = Label_encoder.transform_df(Val_features)
        Test_features = Label_encoder.transform_df(Test_features)
        
        print("Transforming dataframe into numpy arrays")
        #Transforming features df into the x and y for some svm or decision tree
        x_train, y_train = Prepare_data_from_features(Train_features, label_col = "label")
        x_val, y_val = Prepare_data_from_features(Val_features, label_col = "label")
        x_test, y_test = Prepare_data_from_features(Test_features, label_col = "label")
        
        
        print("Scaling the features")
        
        # 1) Fit scaler on train set
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_val_scaled   = scaler.transform(x_val)
        x_test_scaled  = scaler.transform(x_test)
    
    
        print("Performing PCA space reduction")
        #Reducing to PCA
        pca = PCA(n_components=0.99)
        x_train_final = pca.fit_transform(x_train_scaled)
        x_val_final = pca.transform(x_val_scaled)
        x_test_final = pca.transform(x_test_scaled)
        
    
    
        #Data saving and managing
        Processed_data = {
                        'x_train': x_train_final,
                        'y_train': y_train,
                        'x_val': x_val_final,
                        'y_val': y_val,
                        'x_test': x_test_final,
                        'y_test': y_test
                        }
        #Save the file
        with open(PROCESSED_DATA_PATH, 'wb') as f:
            pickle.dump(Processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Scaled and PCA-reduced data saved to disk.")
        
    
    ###################################################################
    # ( 3 ) Training SVM
    ###################################################################
    
    print("\nStarting training the model...")
    
    
    #SVM model
    svc = SVC(C = 1, kernel = 'rbf', degree = 3, gamma = 'scale')
    
    # Fit on training data
    svc.fit(x_train_scaled, y_train)
    
    # Predict
    y_train_pred = svc.predict(x_train_scaled)
    y_val_pred   = svc.predict(x_val_scaled)
    y_test_pred  = svc.predict(x_test_scaled)
    
    # Scores
    
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print("Train acc: ", train_acc)
    print("Val acc: ", val_acc)
    print("Test acc: ", test_acc)











