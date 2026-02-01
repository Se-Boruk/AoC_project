###################################################################
# ( 0 ) Libs and dependencies 
###################################################################

#Basic libs
from comet_ml import Experiment
import sys
import os
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import pickle

from Config import DATABASE_FOLDER, DATASET_PATH, DATASET_NAME, FULL_DATASET_PATH, PROCESSED_DATA_PATH
from Config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_STATE

PCA_n_components = 0.99
SVM_C = 10
SVM_class_weight = 'balanced'
SVM_max_iterations = 5000
Nystroem_n_components = 2000
n_workers = 5

##########################################################################################
#Move folder up to go to database folder to use manager from here
sys.path.insert(0, DATABASE_FOLDER)
from DataBase.DataBase_Functions import Custom_DataSet_Manager, LabelEncoderDF, Prepare_data_from_features
import Functions

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ###################################################################
    # ( 0 ) Comet integration
    ###################################################################
    experiment = Experiment(
        api_key="RoqFxUQ2dJHm8RjW1YatD0VQw",
        project_name="AoC",
        workspace="jbuka"
    )
    experiment.set_name("SVM_Nystroem_C0.1_PCA95")

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
        pca = PCA(n_components=PCA_n_components, random_state=RANDOM_STATE)
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
                        'y_test': y_test,
                        'Label_encoder': Label_encoder
                        }
        #Save the file
        with open(PROCESSED_DATA_PATH, 'wb') as f:
            pickle.dump(Processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Scaled and PCA-reduced data saved to disk.")
        



    ###################################################################
    # ( 2.5 ) Create redced class dataset
    ###################################################################

    #Create reduced class dataset where we merge all classes below 2,5% count into one shared class
    Processed_data_class_reduced = Functions.Drop_Rare_Classes(
        Processed_data, 
        threshold=0.025
    )
    
    ###################################################################
    # ( 3 ) Training SVM (full class)
    ###################################################################
    
    print("\nStarting training the full model...")
    
    model_red = Functions.Train_and_Evaluate_Model(Processed_data=Processed_data,
                                                    model_name="svm_nystroem_rbf.joblib",
                                                    plot_save_name="SVM_full_scores.png",
                                                    suptitle_prefix="SVM full class",
                                                    experiment=experiment
                                                )
    
    print("\nStarting training the full model...")
    model_red = Functions.Train_and_Evaluate_Model(Processed_data= Processed_data_class_reduced,
                                                    model_name="svm_nystroem_rbf_reduced.joblib",
                                                    plot_save_name="SVM_reduced_scores.png",
                                                    suptitle_prefix="SVM class below 2.5% merged",
                                                    experiment=experiment
                                                )








