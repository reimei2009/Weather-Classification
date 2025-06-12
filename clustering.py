import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import tqdm
from config import output_dir

def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, lbls in tqdm(data_loader, desc="Extracting features"):
            images = images.to(device)
            outputs = model(images)
            features.append(outputs.cpu().numpy())
            labels.append(lbls.numpy())
    return np.concatenate(features), np.concatenate(labels)

def perform_clustering(cnn_model, train_loader, num_classes, device, ratio_str):
    train_features, train_labels = extract_features(cnn_model, train_loader, device)
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    n_features = train_features_scaled.shape[1]
    n_components_pca = min(n_features, 20)
    pca = PCA(n_components=n_components_pca)
    train_pca = pca.fit_transform(train_features_scaled)
    n_components_lda = min(num_classes-1, n_features, 20)
    lda = LDA(n_components=n_components_lda)
    train_lda = lda.fit_transform(train_features_scaled, train_labels)
    kmeans_pca = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    pca_labels = kmeans_pca.fit_predict(train_pca)
    pca_silhouette = silhouette_score(train_pca, pca_labels) if len(np.unique(pca_labels)) > 1 else -1
    kmeans_lda = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
    lda_labels = kmeans_lda.fit_predict(train_lda)
    lda_silhouette = silhouette_score(train_lda, lda_labels) if len(np.unique(lda_labels)) > 1 else -1
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(train_pca[:, 0], train_pca[:, 1], c=pca_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'K-Means on PCA (Ratio {ratio_str.replace("_", ":")})')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(train_lda[:, 0], train_lda[:, 1], c=lda_labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'K-Means on LDA (Ratio {ratio_str.replace("_", ":")})')
    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'kmeans_clustering_{ratio_str}.png'))
    plt.show()
    print(f"Silhouette Score (PCA): {pca_silhouette:.4f}")
    print(f"Silhouette Score (LDA): {lda_silhouette:.4f}")
    return {'PCA': {'KMeans': {'silhouette_score': pca_silhouette}}, 'LDA': {'KMeans': {'silhouette_score': lda_silhouette}}}