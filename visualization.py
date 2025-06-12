import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import output_dir

def display_images_per_class(dataset, num_images=5):
    class_names = dataset.classes
    images_per_class = {cls: [] for cls in class_names}
    for img, label in dataset:
        class_name = class_names[label]
        if len(images_per_class[class_name]) < num_images:
            images_per_class[class_name].append(img)
        if all(len(images_per_class[cls]) >= num_images for cls in class_names):
            break
    plt.figure(figsize=(15, len(class_names) * 3))
    for i, cls in enumerate(class_names):
        for j in range(num_images):
            plt.subplot(len(class_names), num_images, i * num_images + j + 1)
            img = images_per_class[cls][j].permute(1, 2, 0).numpy()
            img = img * 0.5 + 0.5
            plt.imshow(img)
            plt.title(cls if j == 0 else "")
            plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_images.png'))
    plt.show()

def plot_pca_lda(train_pca, train_lda, train_labels, class_names, ratio_str):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for label in np.unique(train_labels):
        plt.scatter(train_pca[train_labels == label, 0], train_pca[train_labels == label, 1],
                    label=class_names[label], alpha=0.5)
    plt.title(f'PCA Scatter Plot (Ratio {ratio_str.replace("_", ":")})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    for label in np.unique(train_labels):
        plt.scatter(train_lda[train_labels == label, 0], train_lda[train_labels == label, 1],
                    label=class_names[label], alpha=0.5)
    plt.title(f'LDA Scatter Plot (Ratio {ratio_str.replace("_", ":")})')
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'pca_lda_scatter_{ratio_str}.png'))
    plt.show()

def plot_confusion_matrix(cm, title, class_names):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{title.replace(" ", "_").replace(":", "_")}.png'))
    plt.show()

def compare_within_ratio(results, ratios):
    for ratio in ratios:
        ratio_str = f"ratio_{int(ratio*100)}:{int((1-ratio)*100)}"
        print(f"\n=== So sánh các mô hình trong tỷ lệ {int(ratio*100)}:{int((1-ratio)*100)} ===")
        print(f"{'Model':<20} {'Method':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 60)
        for method in ['PCA', 'LDA']:
            for model_name, metrics in results[ratio_str][method].items():
                print(f"{model_name:<20} {method:<10} {metrics['accuracy']:.4f} {metrics['precision']:.4f} {metrics['recall']:.4f}")
        plt.figure(figsize=(12, 6))
        metrics = ['accuracy', 'precision', 'recall']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, i)
            for method in ['PCA', 'LDA']:
                values = [results[ratio_str][method][model][metric] for model in results[ratio_str][method]]
                plt.bar([f"{model[:10]}\n({method})" for model in results[ratio_str][method]], values, alpha=0.7, label=method)
            plt.title(f'{metric.capitalize()} (Ratio {int(ratio*100)}:{int((1-ratio)*100)})')
            plt.xticks(rotation=45, ha='right')
            plt.ylabel(metric.capitalize())
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_within_ratio_{ratio_str}.png'))
        plt.show()

def compare_across_ratios(results, ratios):
    models = list(results[f"ratio_{int(ratios[0]*100)}:{int((1-ratios[0])*100)}"]['PCA'].keys())
    for model_name in models:
        print(f"\n=== So sánh {model_name} giữa các tỷ lệ ===")
        print(f"{'Ratio':<10} {'Method':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
        print("-" * 50)
        for ratio in ratios:
            ratio_str = f"ratio_{int(ratio*100)}:{int((1-ratio)*100)}"
            for method in ['PCA', 'LDA']:
                metrics = results[ratio_str][method][model_name]
                print(f"{int(ratio*100)}:{int((1-ratio)*100):<10} {method:<10} {metrics['accuracy']:.4f} {metrics['precision']:.4f} {metrics['recall']:.4f}")
        plt.figure(figsize=(12, 6))
        metrics = ['accuracy', 'precision', 'recall']
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, i)
            x = np.arange(len(ratios))
            width = 0.35
            for j, method in enumerate(['PCA', 'LDA']):
                values = [results[f"ratio_{int(r*100)}:{int((1-r)*100)}"][method][model_name][metric] for r in ratios]
                plt.bar(x + j*width, values, width, label=method, alpha=0.7)
            plt.title(f'{metric.capitalize()} ({model_name})')
            plt.xticks(x + width/2, [f"{int(r*100)}:{int((1-r)*100)}" for r in ratios])
            plt.ylabel(metric.capitalize())
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_across_ratios_{model_name.replace(" ", "_")}.png'))
        plt.show()