import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from config import device

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

def evaluate_ml_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'confusion_matrix': cm}

def evaluate_ml_models(cnn_model, train_loader, test_loader, num_classes, device):
    train_features, train_labels = extract_features(cnn_model, train_loader, device)
    test_features, test_labels = extract_features(cnn_model, test_loader, device)
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    pca = PCA(n_components=10)
    train_pca = pca.fit_transform(train_features_scaled)
    test_pca = pca.transform(test_features_scaled)
    lda = LDA(n_components=min(num_classes-1, 10))
    train_lda = lda.fit_transform(train_features_scaled, train_labels)
    test_lda = lda.transform(test_features_scaled)
    knn = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': [3, 5, 7]}, cv=3, n_jobs=1)
    svm = GridSearchCV(SVC(kernel='rbf'), {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}, cv=3, n_jobs=1)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Softmax Regression': LogisticRegression(multi_class='multinomial', max_iter=1000),
        'KNN': knn,
        'SVM': svm,
        'ANN': nn.Sequential(
            nn.Linear(10, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    }
    results = {'PCA': {}, 'LDA': {}}
    for name, model in models.items():
        if name == 'ANN':
            ann_model = models['ANN'].to(device)
            optimizer = torch.optim.Adam(ann_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            X_train_tensor = torch.tensor(train_pca, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(train_labels, dtype=torch.int64).to(device)
            X_test_tensor = torch.tensor(test_pca, dtype=torch.float32).to(device)
            for epoch in range(100):
                ann_model.train()
                optimizer.zero_grad()
                outputs = ann_model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            ann_model.eval()
            with torch.no_grad():
                y_pred = ann_model(X_test_tensor).argmax(dim=1).cpu().numpy()
            acc = accuracy_score(test_labels, y_pred)
            precision = precision_score(test_labels, y_pred, average='weighted', zero_division=0)
            recall = recall_score(test_labels, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(test_labels, y_pred)
            results['PCA'][name] = {'accuracy': acc, 'precision': precision, 'recall': recall, 'confusion_matrix': cm}
        else:
            results['PCA'][name] = evaluate_ml_model(model, train_pca, train_labels, test_pca, test_labels)
    for name, model in models.items():
        if name == 'ANN':
            ann_model = models['ANN'].to(device)
            optimizer = torch.optim.Adam(ann_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            X_train_tensor = torch.tensor(train_lda, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(train_labels, dtype=torch.int64).to(device)
            X_test_tensor = torch.tensor(test_lda, dtype=torch.float32).to(device)
            for epoch in range(100):
                ann_model.train()
                optimizer.zero_grad()
                outputs = ann_model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
            ann_model.eval()
            with torch.no_grad():
                y_pred = ann_model(X_test_tensor).argmax(dim=1).cpu().numpy()
            acc = accuracy_score(test_labels, y_pred)
            precision = precision_score(test_labels, y_pred, average='weighted', zero_division=0)
            recall = recall_score(test_labels, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(test_labels, y_pred)
            results['LDA'][name] = {'accuracy': acc, 'precision': precision, 'recall': recall, 'confusion_matrix': cm}
        else:
            results['LDA'][name] = evaluate_ml_model(model, train_lda, train_labels, test_lda, test_labels)
    return results, train_pca, train_lda, train_labels