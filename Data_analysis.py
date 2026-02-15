###################################################################
# ( 0 ) Libs and dependencies 
###################################################################

#Basic libs
import sys


from Config import DATABASE_FOLDER, DATASET_PATH, DATASET_NAME, FULL_DATASET_PATH
from Config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_STATE



##########################################################################################
#Move folder up to go to database folder to use manager from here
sys.path.insert(0, DATABASE_FOLDER)
from DataBase_Functions import Custom_DataSet_Manager


import numpy as np
import matplotlib.pyplot as plt

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
Train_set, Val_set, Test_set = manager.load_dataset_from_disk(full_dataset_path = FULL_DATASET_PATH)


####################
#histogram of the sample distribution
#####################
Train_class = Train_set['style']
Val_class = Val_set['style']
Test_class = Test_set['style']

#Train data for hist
unique_train, counts_train = np.unique(Train_class, return_counts = True)
normalized_counts_train = counts_train / counts_train.sum()

#Val data for hist
unique_val, counts_val = np.unique(Val_class, return_counts = True)
normalized_counts_val = counts_val/ counts_val.sum()

#Test data for hist
unique_test, counts_test = np.unique(Test_class, return_counts = True)
normalized_counts_test = counts_test / counts_test.sum()


#Managing the order and sorting:
    
sorted_indices = np.argsort(-normalized_counts_train)
ordered_classes = unique_train[sorted_indices]


train_map = dict(zip(unique_train, normalized_counts_train))
val_map = dict(zip(unique_val, normalized_counts_val))
test_map = dict(zip(unique_test, normalized_counts_test))

# Re-generate the values in the exact same order as the sorted Train classes
ordered_train_vals = [train_map.get(c, 0) for c in ordered_classes]
ordered_val_vals = [val_map.get(c, 0) for c in ordered_classes]
ordered_test_vals = [test_map.get(c, 0) for c in ordered_classes]




Ordered_Data_dict = {
    "Train": (ordered_classes, ordered_train_vals),
    "Val": (ordered_classes, ordered_val_vals),
    "Test": (ordered_classes, ordered_test_vals),
}

plt.figure(figsize=(10, 12)) # Set figure size for better vertical spacing

for i, (key, (classes, values)) in enumerate(Ordered_Data_dict.items()):
    plt.subplot(3, 1, i + 1)
    
    plt.bar(classes, values, width=0.8, edgecolor="black", color='skyblue' if key=="Train" else 'lightcoral' if key=="Val" else 'lightgreen')
    
    plt.title(f"Histogram of classes in {key} dataset (Sorted by Train frequency)")
    plt.ylabel("Normalized occurrence")
    plt.tight_layout()
    
    # 1% treshold line
    plt.axhline(y=0.025, color='red', linestyle='--', linewidth=1.5, label='1% Threshold')
    plt.legend()
    
    #nly show on bottom plot labels
    if i == 2:
        plt.xticks(rotation=25, ha='right', fontsize=9)
    else:
        plt.xticks([]) #Hide labels for top and middle plots
    
    plt.grid(axis='y', linestyle='--', alpha=0.7)


plt.tight_layout()

#Save the plot
plot_filename = "WikiArt_Class_Distribution.png"

# Save the figure
plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
plt.show()








