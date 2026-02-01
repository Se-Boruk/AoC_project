from DataBase_Functions import LabelEncoderDF
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

















