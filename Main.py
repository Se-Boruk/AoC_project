###################################################################
# ( 0 ) Libs and dependencies 
###################################################################

#Basic libs
import sys
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from Config import DATABASE_FOLDER, DATASET_PATH, DATASET_NAME
from Config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_STATE



##########################################################################################
#Move folder up to go to database folder to use manager from here
sys.path.insert(0, DATABASE_FOLDER)
from DataBase_Functions import Custom_DataSet_Manager, LabelEncoderDF, Prepare_data_from_features
import Functions




###################################################################
# ( 1 ) Loading data
###################################################################

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
Train_set, Val_set, Test_set = manager.load_dataset_from_disk()



###################################################################
# ( 2 ) Extracting features from image
###################################################################
Train_features, Val_features, Test_features = manager.transform_to_features()


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

print("Performing PCA space reduction")
#Reducing to PCA
pca = PCA(n_components=0.9999)
x_train_p = pca.fit_transform(x_train)
x_val_p = pca.transform(x_val)
x_test_p = pca.transform(x_test)

###################################################################
# ( 3 ) Extracting features from image
###################################################################

print("SVM training")
clf, scaler = Functions.train_svm(x_train_p, y_train,
                                  C= 1.0
                                  )


#######################################
#Evaluation (just realized about unnecessary val set for now. Keeping it to not split sets again)

# Scale all sets using the trained scaler
x_train_scaled = scaler.transform(x_train_p)
x_val_scaled   = scaler.transform(x_val_p)
x_test_scaled  = scaler.transform(x_test_p)

# Predict
y_train_pred = clf.predict(x_train_scaled)
y_val_pred   = clf.predict(x_val_scaled)
y_test_pred  = clf.predict(x_test_scaled)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred)
val_acc   = accuracy_score(y_val, y_val_pred)
test_acc  = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")





