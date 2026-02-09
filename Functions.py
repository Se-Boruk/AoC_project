from DataBase.DataBase_Functions import LabelEncoderDF
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
import Main
from sklearn.multiclass import OneVsRestClassifier
from Config import DATABASE_FOLDER, DATASET_PATH, DATASET_NAME, FULL_DATASET_PATH, PROCESSED_DATA_PATH
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
from PIL import Image
import time
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import cross_val_score
from DataBase import Feature_extraction_functions as Fef


def Drop_Rare_Classes(Processed_data, threshold=0.025):
    """
    Całkowicie usuwa wiersze (obrazy) należące do klas, 
    których liczebność w zbiorze treningowym jest poniżej progu.
    """
    old_encoder = Processed_data['Label_encoder']
    
    # 1. Dekodujemy y_train do nazw, aby policzyć częstość występowania
    y_train_str = pd.Series([old_encoder.int_to_label[i] for i in Processed_data['y_train']])
    
    # 2. Znajdujemy style do zachowania (te powyżej progu)
    style_counts = y_train_str.value_counts(normalize=True)
    valid_styles = style_counts[style_counts >= threshold].index.tolist()
    dropped_styles = style_counts[style_counts < threshold].index.tolist()
    
    print(f"Dropping {len(dropped_styles)} classes (< {threshold*100}%). Keeping {len(valid_styles)} classes.")
    print(f"Dropped styles: {dropped_styles}")

    # 3. Tworzymy nowy Encoder tylko dla "legalnych" stylów
    # Musimy to zrobić, aby etykiety były ciągłe (0, 1, 2...), a nie z dziurami (0, 3, 5...)
    New_Encoder = LabelEncoderDF()
    # Ręcznie ustawiamy mapowanie tylko dla valid_styles
    New_Encoder.label_to_int = {label: i for i, label in enumerate(sorted(valid_styles))}
    New_Encoder.int_to_label = {i: label for label, i in New_Encoder.label_to_int.items()}

    # Funkcja pomocnicza do filtrowania i przeliczania etykiet
    def filter_and_reencode(x_data, y_data):
        # Dekodujemy obecne y do stringów
        y_str = np.array([old_encoder.int_to_label[i] for i in y_data])
        
        # Tworzymy maskę: True tam, gdzie styl jest na liście 'valid_styles'
        mask = np.isin(y_str, valid_styles)
        
        # Filtrujemy X i Y (zostawiamy tylko wiersze True)
        x_filtered = x_data[mask]
        y_str_filtered = y_str[mask]
        
        # Kodujemy y nowym encoderem (do nowych numerków 0..N)
        y_reencoded = np.array([New_Encoder.encode_label(s) for s in y_str_filtered], dtype=np.int64)
        
        return x_filtered, y_reencoded

    # 4. Aplikujemy filtrowanie do wszystkich zbiorów
    print("Filtering Train set...")
    x_train_new, y_train_new = filter_and_reencode(Processed_data['x_train'], Processed_data['y_train'])
    
    print("Filtering Val set...")
    x_val_new, y_val_new = filter_and_reencode(Processed_data['x_val'],   Processed_data['y_val'])
    
    print("Filtering Test set...")
    x_test_new, y_test_new = filter_and_reencode(Processed_data['x_test'],  Processed_data['y_test'])

    # 5. Zwracamy nowy słownik danych
    Processed_data_dropped = {
        'x_train': x_train_new,
        'y_train': y_train_new,
        
        'x_val':   x_val_new,
        'y_val':   y_val_new,
        
        'x_test':  x_test_new,
        'y_test':  y_test_new,
        
        'Label_encoder': New_Encoder
    }
    
    print(f"Dataset filtered. Original Train size: {len(Processed_data['x_train'])}, New Train size: {len(x_train_new)}")
    
    return Processed_data_dropped

def Reduce_Classes_by_Threshold(Processed_data, threshold=0.025):
    """
    Reduces classes based strictly on y_train statistics.
    Carries over all x and y splits to the new dictionary.
    """
    old_encoder = Processed_data['Label_encoder']
    
    # 1. Decode y_train to strings to calculate frequencies
    y_train_str = pd.Series([old_encoder.int_to_label[i] for i in Processed_data['y_train']])
    
    # 2. Identify rare styles based ONLY on training counts
    style_counts = y_train_str.value_counts(normalize=True)
    rare_styles = style_counts[style_counts < threshold].index.tolist()
    
    print(f"Merging {len(rare_styles)} classes (< {threshold*100}% of Train) into 'Other_Styles'...")

    # 3. Initialize the new Encoder (contiguous mapping)
    Reduced_Encoder = LabelEncoderDF()

    def process_split(split_ints):
        """Helper to decode, merge, and re-encode a split."""
        # Convert integers back to strings using the original mapping
        s_str = pd.Series([old_encoder.int_to_label[i] for i in split_ints])
        # Force styles below threshold into the 'Other' bucket
        s_merged = s_str.apply(lambda x: "Other_Styles" if x in rare_styles else x)
        # Map to new contiguous integers (0, 1, 2...)
        return np.array([Reduced_Encoder.encode_label(s) for s in s_merged], dtype=np.int64)

    # 4. Construct the complete dictionary
    # We must explicitly include the 'x' features for val and test
    Processed_data_class_reduced = {
        'x_train': Processed_data['x_train'],
        'y_train': process_split(Processed_data['y_train']),
        
        'x_val':   Processed_data['x_val'],  
        'y_val':   process_split(Processed_data['y_val']),
        
        'x_test':  Processed_data['x_test'],
        'y_test':  process_split(Processed_data['y_test']),
        
        'Label_encoder': Reduced_Encoder
    }

    print(f"Reduction complete. New class count: {len(Reduced_Encoder.label_to_int)}")
    
    return Processed_data_class_reduced




def Train_and_Evaluate_Model(Processed_data,
                             model_name="svm_rbf_nystroem.joblib", 
                             model_folder="models", 
                             plot_save_name="Confusion_Matrix.png", 
                             suptitle_prefix="WikiArt Classification",
                             experiment=None):
    """
    Trains/Loads a Nystroem-approximated SVM and plots Train/Test confusion matrices.
    """
    random_state = 42
    # 1. Setup paths
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_path = os.path.join(model_folder, model_name)
    
    # 2. Persistence: Load or Train
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}...")
        clf = joblib.load(model_path)
    else:
        print(f"\nStarting training the model: {model_name}...")
        
        # SPIRIT: We use Nystroem to approximate the 'RBF' kernel.
        # n_components=300 is a good balance for 57k samples.
        # dual=False is mandatory for n_samples > n_features for speed.
        
        #calculate the % of keeping 
        gamma_val = 1.0 / (Processed_data['x_train'].shape[1] * Processed_data['x_train'].var())
        print("Gamma val: ", gamma_val)
        clf = make_pipeline(
            Nystroem(kernel='rbf', gamma = gamma_val , n_components=Main.Nystroem_n_components, random_state=random_state),
            OneVsRestClassifier(
                LinearSVC(C=Main.SVM_C, class_weight=Main.SVM_class_weight, dual=False, max_iter=Main.SVM_max_iterations),
                n_jobs=Main.n_workers,  # Użyj wszystkich rdzeni!
            ))
        if experiment:
            params = {
                "model_type": "LinearSVC (Optimized)",
                "n_components_pca": Main.PCA_n_components, # Wartość z Main.py (możesz przekazać jako arg)
                "svm_C": Main.SVM_C,             # To co ustawiłeś w LinearSVC
                "nystroem_components": Main.Nystroem_n_components,
                "class_weight": Main.SVM_class_weight
            }
            experiment.log_parameters(params)

        # Fit on training data
        clf.fit(Processed_data['x_train'], Processed_data['y_train'])
        
        # Save model
        joblib.dump(clf, model_path)
        
        # Podmieniamy zmienną clf na pipeline, żeby kod poniżej (macierze pomyłek) zadziałał
        print(f"Model saved to {model_path}")

    # 3. Predict and Scores
    y_train_pred = clf.predict(Processed_data['x_train'])
    y_test_pred  = clf.predict(Processed_data['x_test'])
    
    train_acc = accuracy_score(Processed_data['y_train'], y_train_pred)
    test_acc  = accuracy_score(Processed_data['y_test'], y_test_pred)
    
    print(f"\n{suptitle_prefix} Results:")
    print(f"Train acc: {train_acc:.4f}")
    print(f"Test acc:  {test_acc:.4f}")

    if experiment:
        experiment.log_metrics({
            "train_accuracy": train_acc,
            "test_accuracy": test_acc
        })

    # 4. Visualization
    Label_encoder = Processed_data["Label_encoder"]
    
    print("\nGenerating Confusion Matrices...")
    class_names = [Label_encoder.int_to_label[i] for i in range(len(Label_encoder.int_to_label))]

    fig, axes = plt.subplots(1, 2, figsize=(28, 14))
    
    datasets = [
        ("Train", Processed_data['y_train'], y_train_pred, train_acc),
        ("Test", Processed_data['y_test'], y_test_pred, test_acc)
    ]

    for i, (name, y_true, y_pred, acc) in enumerate(datasets):
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=axes[i], cmap='viridis', xticks_rotation=90, values_format='d')
        axes[i].set_title(f"Confusion Matrix: {name} Set\nAccuracy: {acc:.2%}")

    # Steerable suptitle
    fig.suptitle(f"{suptitle_prefix}\n(Overall Test Accuracy: {test_acc:.2%})", fontsize=22, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_save_name, dpi=100, bbox_inches='tight')
    if experiment:
        # Wysyłamy wykres do Cometa
        experiment.log_figure(figure_name=suptitle_prefix, figure=fig)
    
    print(f"Plot saved to: {plot_save_name}")
    plt.show()
    
    return clf

def Predict_Single_Image(Processed_data, image_features, model_path = "models/svm_nystroem_rbf.joblib"):
    """
    Function predicts label for one single image

    !!! image_features need to be processed by PCA !!!
    """
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} does not exist")
    else:
        print(f"Loading model {model_path} from disk...")

    Label_encoder = Processed_data["Label_encoder"]

    clf = joblib.load(model_path)

    if len(image_features.shape) == 1:
        image_features = image_features.reshape(1, -1)

    prediction_int = clf.predict(image_features)[0]

    
    class_name = Label_encoder.int_to_label[prediction_int]
    return class_name, prediction_int


def Train_KMeans(x_train, n_clusters=50):
    """Trenuje model raz i zwraca go wraz z etykietami."""

    model_path = "models/Kmeans_learned_test.joblib"
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from Kmeans_learned_test...")
        kmeans = joblib.load(model_path)
        train_cluster_assignments = kmeans.labels_
    else:    
        print(f"Trenowanie KMeans na {len(x_train)} próbkach...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        train_cluster_assignments = kmeans.fit_predict(x_train)
        joblib.dump(kmeans, model_path)

    return kmeans, train_cluster_assignments


def Get_5_Most_Similar_Images(new_image_vector, x_train, kmeans_model, train_cluster_assignments):
    """
    new_image_vector: wektor po PCA o kształcie (1, n_components)
    x_train: baza wektorów treningowych po PCA
    train_cluster_assignments: wynik kmeans.labels_ dla x_train
    """
    if len(new_image_vector.shape) == 1:
        new_image_vector = new_image_vector.reshape(1, -1)

    # 1. Sprawdź, do którego klastra trafia nowy obraz
    cluster_id = kmeans_model.predict(new_image_vector)[0]
    
    # 2. Wyciągnij wszystkie obrazy z tego samego klastra
    in_cluster_indices = np.where(train_cluster_assignments == cluster_id)[0]
    cluster_vectors = x_train[in_cluster_indices]
    
    # 3. Oblicz odległość euklidesową od Twojego obrazu do wszystkich w klastrze
    distances = euclidean_distances(new_image_vector, cluster_vectors).flatten()
    
    # 4. Pobierz indeksy 5 najmniejszych odległości
    # (argsorted sortuje od najmniejszej odległości)
    closest_relative_indices = np.argsort(distances)[:5]
    
    # Mapowanie z powrotem na oryginalne indeksy x_train
    original_indices = in_cluster_indices[closest_relative_indices]
    
    return original_indices, distances[closest_relative_indices]



def Plot_Similar_Images(target_path, similar_indices, Train_set):
    """
    target_path: ścieżka do obrazu zapytania
    similar_indices: indeksy zwrócone przez funkcję Get_5_Most_Similar_Images
    Train_set: Twój DataFrame z kolumną 'path' i 'label'
    """
    fig, axes = plt.subplots(1, 6, figsize=(20, 5))
    
    # 1. Obraz oryginalny (Query)
    axes[0].imshow(Image.open(target_path))
    axes[0].set_title("ORYGINAŁ")
    axes[0].axis('off')
    
    # 2. Obrazy podobne
    for i, idx in enumerate(similar_indices):
        path = Train_set.iloc[idx]['path']
        label = Train_set.iloc[idx]['style']
        
        axes[i+1].imshow(Image.open(path))
        axes[i+1].set_title(f"Podobny {i+1}\n({label})")
        axes[i+1].axis('off')
        
    plt.tight_layout()
    plt.show()

def Analyze_Prediction_Hierarchy(model, image_features, true_label_int, encoder):
    """
    Pokazuje, które miejsce w rankingu modelu zajęła prawidłowa etykieta.
    """
    # 1. Obliczamy wyniki (decision scores) dla wszystkich klas
    # image_features musi mieć kształt (1, n_features)
    if len(image_features.shape) == 1:
        image_features = image_features.reshape(1, -1)
        
    scores = model.decision_function(image_features)[0]
    
    # 2. Sortujemy indeksy klas według wyników (malejąco)
    ranked_indices = np.argsort(scores)[::-1]
    
    # 3. Znajdujemy pozycję prawidłowej etykiety
    # +1, bo indeksy w Pythonie zaczynają się od 0
    rank_position = np.where(ranked_indices == true_label_int)[0][0] + 1
    
    # 4. Wyciągamy nazwy stylów dla topowych typowań
    top_3_indices = ranked_indices[:3]
    top_3_styles = [encoder.int_to_label[i] for i in top_3_indices]
    
    true_style_name = encoder.int_to_label[true_label_int]
    predicted_style_name = encoder.int_to_label[ranked_indices[0]]

    print(f"--- ANALIZA HIERARCHII ---")
    print(f"Prawidłowy styl: {true_style_name}")
    print(f"Typowanie modelu (Top 1): {predicted_style_name}")
    print(f"Miejsce prawidłowego stylu w rankingu: {rank_position} / {len(scores)}")
    print(f"Top 3 wybory modelu: {top_3_styles}")
    
    return rank_position

def Evaluate_Hierarchy_On_Full_Set(model, x_data, y_data, encoder, plot_title="Model Ranking Analysis"):
    """
    Analizuje skuteczność modelu na całym zbiorze, sprawdzając, 
    na którym miejscu w rankingu znalazła się poprawna etykieta.
    
    Zwraca: słownik z metrykami (Top-1, Top-3, Top-5, Mean Rank)
    """
    print(f"Obliczanie hierarchii predykcji dla {len(x_data)} próbek...")
    
    # 1. Pobieramy wyniki decyzyjne dla całego zbioru naraz (Macierz: N_samples x N_classes)
    # To jest znacznie szybsze niż pętla for
    all_scores = model.decision_function(x_data)
    
    # 2. Sortujemy indeksy klas od najbardziej prawdopodobnych (malejąco)
    # axis=1 oznacza sortowanie wierszami (dla każdego obrazu osobno)
    predicted_rankings = np.argsort(all_scores, axis=1)[:, ::-1]
    
    # 3. Znajdujemy pozycję (rank) prawdziwej etykiety dla każdego obrazu
    # Tworzymy listę, gdzie 1 = idealnie trafione, 2 = było na drugim miejscu itd.
    true_ranks = []
    
    # Optymalizacja: Pętla jest tu akceptowalna, ale dla super wydajności można użyć broadcastingu
    for i, true_label in enumerate(y_data):
        # np.where zwraca tuplę, bierzemy [0][0] żeby dostać indeks
        # Dodajemy +1, żeby ranking był liczony od 1 (a nie od 0)
        rank = np.where(predicted_rankings[i] == true_label)[0][0] + 1
        true_ranks.append(rank)
        
    true_ranks = np.array(true_ranks)
    
    # 4. Obliczamy metryki
    top_1_acc = np.mean(true_ranks == 1)
    top_3_acc = np.mean(true_ranks <= 3)
    top_5_acc = np.mean(true_ranks <= 5)
    mean_rank = np.mean(true_ranks)
    median_rank = np.median(true_ranks)
    
    print(f"\n--- WYNIKI HIERARCHICZNE ({plot_title}) ---")
    print(f"Top-1 Accuracy (Standard): {top_1_acc:.2%}")
    print(f"Top-3 Accuracy:            {top_3_acc:.2%}")
    print(f"Top-5 Accuracy:            {top_5_acc:.2%}")
    print(f"Średnia pozycja poprawnej etykiety: {mean_rank:.2f}")
    print(f"Mediana pozycji: {median_rank}")
    
    # 5. Wizualizacja
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Wykres A: Porównanie Top-N
    metrics = ['Top-1', 'Top-3', 'Top-5']
    values = [top_1_acc, top_3_acc, top_5_acc]
    bars = ax[0].bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax[0].set_ylim(0, 1.0)
    ax[0].set_title("Skuteczność Top-N")
    ax[0].bar_label(bars, fmt='{:.1%}')
    
    # Wykres B: Histogram pozycji (Rank Distribution)
    # Ocinamy histogram do 20 pierwszych miejsc dla czytelności
    ax[1].hist(true_ranks, bins=range(1, 21), color='purple', alpha=0.7, edgecolor='black', align='left')
    ax[1].set_xticks(range(1, 21))
    ax[1].set_xlabel("Pozycja poprawnej etykiety w rankingu modelu")
    ax[1].set_ylabel("Liczba obrazów")
    ax[1].set_title("Rozkład błędów (Gdzie ląduje prawda?)")
    
    plt.suptitle(plot_title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return {
        "top1": top_1_acc,
        "top3": top_3_acc,
        "top5": top_5_acc,
        "mean_rank": mean_rank,
        "ranks_array": true_ranks
    }

def Run_RFE_Selection(x_train, y_train, min_features_to_select=500, step=0.1, n_jobs = 5):
    """
    x_train: Zredukowany zbiór treningowy (np. te 10%)
    n_features_to_keep: Ile cech chcemy zostawić na koniec
    step: Ile cech usuwać w każdym kroku (float = procent, int = liczba)
    """
    print(f"Rozpoczynam RFE. Startowa liczba cech: {x_train.shape[1]}")
    
    # Używamy LinearSVC, bo jest szybki i ma atrybut .coef_
    # dual=False jest szybsze gdy n_samples > n_features
    svm = LinearSVC(C=0.1, max_iter=2000, dual=False, class_weight='balanced', tol=1e-2, random_state=42)
    
    # Konfiguracja RFE
    rfe = RFECV(estimator=svm, 
              min_features_to_select=min_features_to_select, 
              step=step,
              cv=3,
              n_jobs= n_jobs, 
              scoring='accuracy',
              verbose=1) # verbose=1 pokaże postęp
    
    start_time = time.time()
    rfe.fit(x_train, y_train)
    end_time = time.time()
    
    print(f"RFE zakończone w {(end_time - start_time)/60:.2f} minut.")
    
    return rfe

def Run_Group_RFE(x_train_dict, y_train, cv=3, n_jobs=5):
    """
    Wykonuje Recursive Feature Elimination na CAŁYCH ekstraktorach cech.
    
    x_train_dict: słownik postaci {'nazwa_filtra': numpy_array, 'inny_filtr': numpy_array}
                  Wymiary każdego array: (n_samples, n_features_filtra)
    """
    # Kopiujemy listę kluczy (aktywne ekstraktory), żeby móc z niej usuwać
    active_extractors = list(x_train_dict.keys())
    elimination_history = []
    
    print(f"Rozpoczynam Group RFE dla ekstraktorów: {active_extractors}")
    start_time = time.time()
    
    while len(active_extractors) > 0:
        # 1. Zbuduj połączoną macierz X z aktualnie aktywnych ekstraktorów
        X_list = []
        feature_slices = {} # Słownik pamiętający: filtr -> (start_col, end_col)
        current_col = 0
        
        for name in active_extractors:
            features = x_train_dict[name]
            # Upewnienie się, że to macierz 2D
            if len(features.shape) == 1:
                features = features.reshape(-1, 1)
                
            X_list.append(features)
            num_cols = features.shape[1]
            feature_slices[name] = (current_col, current_col + num_cols)
            current_col += num_cols
            
        X_concat = np.hstack(X_list)
        
        # 2. Trenuj model i sprawdź dokładność na Cross-Validation (żeby widzieć spadek/wzrost)
        svm = LinearSVC(C=0.1, max_iter=2000, dual=False, class_weight='balanced', tol=1e-2, random_state=42)
        print(f"\nTrenowanie SVM na {X_concat.shape[1]} cechach z {len(active_extractors)} ekstraktorów...")
        
        scores = cross_val_score(svm, X_concat, y_train, cv=cv, scoring='accuracy', n_jobs=n_jobs)
        mean_acc = np.mean(scores)
        
        # Jeśli to ostatni ekstraktor, zapisz go i zakończ
        if len(active_extractors) == 1:
            print(f"Ostatni pozostały: {active_extractors[0]} | Dokładność CV: {mean_acc:.4f}")
            elimination_history.append({
                'dropped': active_extractors[0],
                'accuracy': mean_acc,
                'reason': 'Zwycięzca (ostatni na placu boju)'
            })
            break
            
        # 3. Dopasuj model na całych danych, aby uzyskać wagi (.coef_)
        svm.fit(X_concat, y_train)
        
        # .coef_ w OneVsRest dla wielu klas ma kształt (n_classes, n_features)
        # Bierzemy wartość bezwzględną (ważność) i uśredniamy dla wszystkich klas
        importances = np.mean(np.abs(svm.coef_), axis=0)
        
        # 4. Oblicz "siłę" (wynik) każdego ekstraktora
        extractor_scores = {}
        for name in active_extractors:
            start, end = feature_slices[name]
            # Bierzemy średnią z wag wszystkich cech z danego ekstraktora
            # (Średnia jest lepsza niż suma, by małe ekstraktory miały równe szanse z wielkimi)
            extractor_scores[name] = np.mean(importances[start:end])
            
        # 5. Znajdź i usuń najsłabszy ekstraktor
        worst_extractor = min(extractor_scores, key=extractor_scores.get)
        
        elimination_history.append({
            'dropped': worst_extractor,
            'accuracy': mean_acc,
            'score': extractor_scores[worst_extractor]
        })
        
        print(f"Dokładność CV zbioru: {mean_acc:.4f}")
        print(f" => USUWAM NAJSŁABSZY EKSTRAKTOR: {worst_extractor} (Średnia waga: {extractor_scores[worst_extractor]:.6f})")
        
        active_extractors.remove(worst_extractor)

    end_time = time.time()
    
    # 6. Podsumowanie - tworzymy Ranking
    print("\n" + "="*50)
    print("RANKING EKSTRAKTORÓW (Od NAJLEPSZEGO do NAJGORSZEGO)")
    print("="*50)
    
    # Odwracamy listę usuniętych, bo ten usunięty na końcu jest najlepszy
    ranked_extractors = [item['dropped'] for item in reversed(elimination_history)]
    for i, name in enumerate(ranked_extractors, 1):
        print(f"{i}. {name}")
        
    print(f"\nCzas trwania analizy: {(end_time - start_time)/60:.2f} minut.")
    
    return ranked_extractors, elimination_history

def Extract_image_features_RFE(row_tuple, bovw_manager=None):
    """
    Worker function to process one image.
    Zwraca słownik zagnieżdżony: { 'grupa': {cechy}, 'inna_grupa': [tablica], ... }
    """
    _, row = row_tuple
    try:
        with Image.open(row['path']) as img_raw:
            # Konwersja i resize (bez zmian)
            img_rgb = img_raw.convert('RGB')
            img_rgb.thumbnail((512, 512), Image.Resampling.BILINEAR)
            img = np.array(img_rgb, dtype=np.uint8)

        # Główny słownik - teraz będzie przechowywał GRUPY cech
        # 'label' zostawiamy na wierzchu, bo to y_train
        grouped_features = {'label': row['style']}

        # --- SEKCJA EKSTRAKCJI ---
        # Zamiast .update(), przypisujemy do KLUCZY (nazw grup)

        if bovw_manager is not None:
            # BoVW zazwyczaj zwraca histogram (array lub dict)
            grouped_features['BoVW'] = bovw_manager.compute_bovw_histogram(img)

        # Pojedyncze wartości też pakujemy
        grouped_features['Entropy'] = Fef.img_entropy(img, n_bins=64)

        # Funkcje zwracające słowniki lub tablice
        grouped_features['Color_Moments'] = Fef.color_moments(img)
        grouped_features['Color_Saturation'] = Fef.colorfulness_saturation(img)
        grouped_features['Color_Harmony'] = Fef.color_harmony_contrast(img)
        grouped_features['LAB_Histogram'] = Fef.lab_histogram(img)
        
        grouped_features['Haralick'] = Fef.haralick_features(img)
        grouped_features['Gabor'] = Fef.gabor_features(img)

        # LBP często zwraca czystą tablicę numpy
        grouped_features['LBP'] = Fef.lbp_histogram(img)

        grouped_features['Wavelet'] = Fef.wavelet_texture(img)
        grouped_features['Edge_Stats'] = Fef.edge_statistics(img)
        
        grouped_features['Radial_Spectral'] = Fef.radial_spectral_summary(img)
        grouped_features['Auto_Correlogram'] = Fef.color_auto_correlogram(img)
        
        grouped_features['HOG_Statistics'] = Fef.hog_stats(img)

        grouped_features['Depth_of_field'] = Fef.depth_of_field_proxy(img) # Głębia ostrości
        grouped_features['Rule_of_thirds_stats'] = Fef.rule_of_thirds_stats(img)

        return grouped_features
    
    except Exception as e:
        print(f"Error processing {row['path']}: {e}")
        return None










