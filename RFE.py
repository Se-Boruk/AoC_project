import pandas as pd
import numpy as np
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
# Import twoich modułów
import Functions
import DataBase.Feature_extraction_functions as Fef

from DataBase.DataBase_Functions import Custom_DataSet_Manager, LabelEncoderDF
from Config import DATASET_PATH, DATASET_NAME, FULL_DATASET_PATH
from Config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT, RANDOM_STATE

###################################################################################
# KROK 0: Deklaracja niezbędnych funkcji
###################################################################################
def run_rfe_extraction(df):
    cores = os.cpu_count()
    # Ustawiamy funkcję worker z parametrem bovw
    worker = partial(Functions.Extract_image_features_RFE, bovw_manager=bovw_manager)
    
    with ProcessPoolExecutor(max_workers=cores) as executor:
        results = list(tqdm(
            executor.map(worker, df.iterrows(), chunksize=32),
            total=len(df),
            desc="Extracting Features for RFE"
        ))
    return [r for r in results if r is not None]

def Convert_List_To_Dict_Of_Arrays(features_list):
    """
    Zamienia listę [{'hog': [...], 'lbp': [...]}, ...] 
    na {'hog': np.array(N, D), 'lbp': np.array(N, D)}
    """
    if not features_list: return None, None
    
    # 1. Wyciągamy etykiety
    y_labels = [d['label'] for d in features_list]
    
    # 2. Pobieramy nazwy dostępnych grup (kluczy) z pierwszego elementu
    # (pomijamy 'label')
    keys = [k for k in features_list[0].keys() if k != 'label']
    
    x_dict = {}
    print("Konwersja danych do formatu macierzowego...")
    
    for key in keys:
        # Zbieramy dane dla tego klucza ze wszystkich obrazów
        column_data = []
        for item in features_list:
            val = item[key]
            # Spłaszczamy, jeśli to dict lub lista list
            if isinstance(val, dict):
                val = np.array(list(val.values())).flatten()
            elif isinstance(val, (list, tuple, np.ndarray)):
                val = np.array(val).flatten()
            else: # pojedyncza liczba
                val = np.array([val])
                
            column_data.append(val)
            
        # Tworzymy macierz numpy
        x_dict[key] = np.array(column_data, dtype=np.float32)
        print(f" -> Grupa '{key}': {x_dict[key].shape}")
        
    return x_dict, np.array(y_labels)


if __name__ == '__main__':  
###################################################################################
# KROK 1: Wybór 10% danych (STRATIFIED)
###################################################################################

    manager = Custom_DataSet_Manager(DataSet_path = DATASET_PATH,
                                         train_split = TRAIN_SPLIT,
                                         val_split = VAL_SPLIT,
                                         test_split = TEST_SPLIT,
                                         random_state = RANDOM_STATE
                                         )
    
    #Download data if not present
    manager.download_database(DATASET_NAME)
    Train_set, Val_set, Test_set = manager.load_dataset_from_disk(full_dataset_path = FULL_DATASET_PATH)
            

    print("Losowanie 10% zbioru treningowego do analizy RFE...")
    # Zakładam, że manager jest już zainicjalizowany jako 'manager'
    # Używamy stratify, żeby zachować proporcje stylów
    df_small_train, _ = train_test_split(
        Train_set, 
        train_size=0.10, 
        stratify=Train_set['style'], 
        random_state=42
    )

    print(f"Wybrano {len(df_small_train)} obrazów do analizy.")

###################################################################################
# KROK 2: Przygotowanie BoVW Manager (musi być wytrenowany!)
###################################################################################
    bovw_manager = Fef.BoVW_Manager(n_clusters=500)
    bovw_path = "models/bovw_vocab_RBF.pkl"

    if os.path.exists(bovw_path):
        print("Loading BoVW vocabulary...")
        bovw_manager.load_vocab(bovw_path)
    else:
        # Jeśli nie masz wytrenowanego, musisz go wytrenować (możesz użyć do tego df_small_train)
        print("Training BoVW vocabulary on small subset...")
        paths = [os.path.join(FULL_DATASET_PATH, p) for p in df_small_train['path']]
        bovw_manager.fit_vocabulary(paths, sample_size=2000)
        bovw_manager.save_vocab(bovw_path)

###################################################################################
# KROK 3: Ekstrakcja cech w formacie GRUPOWYM
###################################################################################

    # Uruchamiamy ekstrakcję
    features_list = run_rfe_extraction(df_small_train)

    # Konwertujemy
    x_train_dict, y_train_raw = Convert_List_To_Dict_Of_Arrays(features_list)

    ###################################################################################
    # KROK 5: Label Encoding i Skalowanie
    ###################################################################################
    # Musimy zakodować stringi 'style' na inty

    encoder = LabelEncoderDF()
    # Tworzymy DF tylko po to, żeby użyć Twojego encodera (lub zrób to ręcznie)
    temp_df = pd.DataFrame({'label': y_train_raw})
    temp_df = encoder.transform_df(temp_df, 'label')
    y_train_encoded = temp_df['label'].values

    # Skalowanie (StandardScaler) dla KAŻDEJ grupy osobno
    # To ważne, żeby SVM nie faworyzował grup o dużych liczbach
    scaler_dict = {}
    x_train_scaled_dict = {}

    print("Skalowanie cech...")
    for key, matrix in x_train_dict.items():
        scaler = StandardScaler()
        # Zastępujemy NaNy zerami (bezpiecznik)
        matrix = np.nan_to_num(matrix)
        x_train_scaled_dict[key] = scaler.fit_transform(matrix)

    ###################################################################################
    # KROK 6: Uruchomienie Group RFE
    ###################################################################################

    # Teraz masz wszystko gotowe do funkcji, którą masz w Functions.py
    ranked_extractors, history = Functions.Run_Group_RFE(
        x_train_dict=x_train_scaled_dict, 
        y_train=y_train_encoded, 
        cv=3, 
        n_jobs=6 # Dostosuj do swojego CPU
    )

    # Opcjonalnie: Zapisz wyniki
    
    joblib.dump(history, "models/group_rfe_results.pkl")