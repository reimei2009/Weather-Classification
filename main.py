import os
import torch
from torchvision import transforms
from config import ratios, num_epochs, lr, device, output_dir
from data_loader import load_dataset, get_data_loaders
from model import AdvancedWeatherCNN
from train import fit, get_latest_epoch
from visualization import display_images_per_class, plot_pca_lda, plot_confusion_matrix, compare_within_ratio, compare_across_ratios
from ml_evaluation import evaluate_ml_models
from clustering import perform_clustering

def main():
    # Load dataset
    full_dataset = load_dataset()
    num_classes = len(full_dataset.classes)
    class_names = full_dataset.classes

    # Display sample images
    raw_dataset = ImageFolder('/kaggle/input/weather-dataset/dataset', transform=transforms.ToTensor())
    display_images_per_class(raw_dataset)

    # Train and evaluate CNN for each ratio
    histories = {}
    for ratio in ratios:
        ratio_str = f"{int(ratio*100)}_{int((1-ratio)*100)}"
        prefix = f"model_epoch_{ratio_str}_"
        print(f"\n=== Training with train-test ratio {int(ratio*100)}:{int((1-ratio)*100)} ===")
        start_epoch, latest_model = get_latest_epoch(prefix=prefix)
        cnn_model = AdvancedWeatherCNN(num_classes).to(device)
        if latest_model and start_epoch > 0:
            model_path = os.path.join(output_dir, latest_model)
            if torch.cuda.is_available():
                cnn_model.load_state_dict(torch.load(model_path, weights_only=True))
            else:
                cnn_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
            print(f"Resuming from epoch {start_epoch} with model {model_path}")
        else:
            print("Training from scratch.")
            start_epoch = 0
        train_loader, test_loader = get_data_loaders(full_dataset, ratio)
        h = fit(
            num_epochs, lr, cnn_model, train_loader, test_loader, device,
            start_epoch=start_epoch,
            loss_save_path=os.path.join(output_dir, f'best_loss_model_{ratio_str}.pth'),
            acc_save_path=os.path.join(output_dir, f'best_acc_model_{ratio_str}.pth'),
            prefix=prefix
        )
        histories[f'ratio_{ratio_str}'] = h

    # ML evaluation and visualization
    results = {}
    for ratio in ratios:
        ratio_str = f"ratio_{int(ratio*100)}:{int((1-ratio)*100)}"
        print(f"\n=== Evaluating ML models for train-test ratio {int(ratio*100)}:{int((1-ratio)*100)} ===")
        train_loader, test_loader = get_data_loaders(full_dataset, ratio)
        result, train_pca, train_lda, train_labels = evaluate_ml_models(cnn_model, train_loader, test_loader, num_classes, device)
        results[ratio_str] = result
        plot_pca_lda(train_pca, train_lda, train_labels, class_names, ratio_str)
        for method in ['PCA', 'LDA']:
            for model_name in result[method]:
                cm = result[method][model_name]['confusion_matrix']
                title = f'{model_name} ({method}, Ratio {int(ratio*100)}:{int((1-ratio)*100)})'
                plot_confusion_matrix(cm, title, class_names)

    # Compare ML models
    compare_within_ratio(results, ratios)
    compare_across_ratios(results, ratios)

    # Clustering
    clustering_results = {}
    for ratio in ratios:
        ratio_str = f"ratio_{int(ratio*100)}:{int((1-ratio)*100)}"
        print(f"\n=== K-Means Clustering for train-test ratio {int(ratio*100)}:{int((1-ratio)*100)} ===")
        train_loader, _ = get_data_loaders(full_dataset, ratio)
        clustering_results[ratio_str] = perform_clustering(cnn_model, train_loader, num_classes, device, ratio_str)

    # Print clustering summary
    print("\n=== Silhouette Scores for K-Means Clustering ===")
    print(f"{'Ratio':<10} {'Method':<10} {'Silhouette Score':<15}")
    print("-" * 35)
    for ratio in ratios:
        ratio_str = f"ratio_{int(ratio*100)}:{int((1-ratio)*100)}"
        for method in ['PCA', 'LDA']:
            score = clustering_results[ratio_str][method]['KMeans']['silhouette_score']
            print(f"{int(ratio*100)}:{int((1-ratio)*100):<10} {method:<10} {score:.4f}")

if __name__ == "__main__":
    main()