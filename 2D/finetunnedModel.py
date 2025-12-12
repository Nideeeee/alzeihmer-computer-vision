import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report
from sklearn.manifold import TSNE
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import umap


class BaseAlzheimerDataset(Dataset):
    # Normalisation pour images en niveaux de gris (1 canal)
    # Moyenne des valeurs ImageNet RGB: (0.485 + 0.456 + 0.406) / 3 ≈ 0.449
    # Moyenne des std ImageNet RGB: (0.229 + 0.224 + 0.225) / 3 ≈ 0.226
    GRAYSCALE_MEAN = torch.tensor([0.449]).view(1, 1, 1)
    GRAYSCALE_STD = torch.tensor([0.226]).view(1, 1, 1)

    def _extract_subject_id(self, filename):
        parts = filename.split('_')
        for i in range(len(parts) - 2):
            if parts[i+1] == 'S':
                return f"{parts[i]}_{parts[i+1]}_{parts[i+2]}"
        return None

    def _preprocess_image(self, img_path):
        image = np.load(img_path)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = np.expand_dims(image, axis=0)  # dimension = (1, H, W)
        image = torch.FloatTensor(image)

        if image.shape[1] != 224 or image.shape[2] != 224:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

        return image

    def __len__(self):
        return len(self.samples)


class AlzheimerTrainDataset(BaseAlzheimerDataset):
    def __init__(self, data_dir, csv_file, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.df = pd.read_csv(csv_file, header=None)
        self.df.columns = ['subject_id', 'id', 'age', 'gender', 'diagnosis',
                           'mci_level', 'col6', 'col7', 'col8', 'col9', 'col10']
        self.samples = []
        
        stats = {'CN': 0, 'AD': 0}
        for npy_file in self.data_dir.glob("*.npy"):
            subject_id = self._extract_subject_id(npy_file.stem)
            if subject_id:
                match = self.df[self.df['subject_id'] == subject_id]
                if not match.empty:
                    diagnosis = match.iloc[0]['diagnosis']
                    if diagnosis == 'CN':
                        self.samples.append((npy_file, 0))
                        stats['CN'] += 1
                    elif diagnosis == 'AD':
                        self.samples.append((npy_file, 1))
                        stats['AD'] += 1

        print(f"Charge {len(self.samples)} images (CN: {stats['CN']}, AD: {stats['AD']})")

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = self._preprocess_image(img_path)

        if self.transform:
            image = self.transform(image)

        image = (image - self.GRAYSCALE_MEAN) / self.GRAYSCALE_STD
        return image, label


class AlzheimerValidDataset(BaseAlzheimerDataset):
    def __init__(self, data_dir, csv_file):
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(csv_file, header=None)
        self.df.columns = ['subject_id', 'id', 'age', 'gender', 'diagnosis',
                           'mci_level', 'col6', 'col7', 'col8', 'col9', 'col10']
        self.samples = []
        
        stats = {'MCI-3': 0, 'MCI-4': 0}
        for npy_file in self.data_dir.glob("*.npy"):
            subject_id = self._extract_subject_id(npy_file.stem)
            if subject_id:
                match = self.df[self.df['subject_id'] == subject_id]
                if not match.empty:
                    diagnosis = match.iloc[0]['diagnosis']
                    mci_level = match.iloc[0]['mci_level']
                    if diagnosis == 'MCI':
                        if mci_level == 3:
                            self.samples.append((npy_file, 1))
                            stats['MCI-3'] += 1
                        elif mci_level == 4:
                            self.samples.append((npy_file, 0))
                            stats['MCI-4'] += 1

        print(f"Charge {len(self.samples)} images MCI (MCI-3: {stats['MCI-3']}, MCI-4: {stats['MCI-4']})")

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = self._preprocess_image(img_path)
        image = (image - self.GRAYSCALE_MEAN) / self.GRAYSCALE_STD
        return image, label


class CNN2DModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Adapter conv1 pour accepter 1 canal (niveaux de gris) au lieu de 3
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=1,      # 1 canal au lieu de 3 (nouveauté après soutenance)
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        with torch.no_grad():
            self.resnet.conv1.weight = nn.Parameter(
                original_conv1.weight.mean(dim=1, keepdim=True)
            )

        #on geler les premieres couches (SAUF conv1 qui doit s'adapter aux niveaux de gris)
        for name, param in self.resnet.named_parameters():
            if name.startswith(('bn1', 'layer1')):
                param.requires_grad = False
        # conv1 reste entraînable pour s'adapter aux images en niveaux de gris
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)
    
    def extract_features(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc[0](x)  # Dropout
        x = self.resnet.fc[1](x)  # Linear -> 256
        x = self.resnet.fc[2](x)  # ReLU
        x = self.resnet.fc[3](x)  # BatchNorm
        # On s'arrête ici pour avoir les embeddings de dimension 256
        return x
    
    def classify_from_embedding(self, embedding):
        x = self.resnet.fc[4](embedding)  # Dropout
        x = self.resnet.fc[5](x)          # Linear(256, 2)
        return x


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in tqdm(dataloader, desc='Train'):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / len(dataloader), accuracy_score(all_labels, all_preds)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Valid'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / len(dataloader), accuracy_score(all_labels, all_preds), all_preds, all_labels


def extract_embeddings(model, dataloader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extraction'):
            images = images.to(device)
            # Extraire les features avant la dernière couche
            features = model.extract_features(images)
            all_embeddings.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    embeddings = np.vstack(all_embeddings)
    labels = np.array(all_labels)
    return embeddings, labels


def extract_embeddings_per_patient(model, dataset, device, batch_size=16):
    model.eval()
    patient_data = {}
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embeddings = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Extraction'):
            images = images.to(device)
            features = model.extract_features(images)
            all_embeddings.append(features.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    
    # maintenant on regroupe par patient
    for idx, (npy_path, label) in enumerate(dataset.samples):
        subject_id = dataset._extract_subject_id(npy_path.stem)
        
        if subject_id is None:
            print(f"Warning: impossible d'extraire subject_id de {npy_path.name}")
            continue
        
        if subject_id not in patient_data:
            patient_data[subject_id] = {
                'embeddings': [],
                'label': label
            }
        
        patient_data[subject_id]['embeddings'].append(all_embeddings[idx])
    
    #moyenne des embeddings par patient
    patient_embeddings = []
    patient_labels = []
    patient_ids = []
    
    for subject_id, data in patient_data.items():
        stacked = torch.stack(data['embeddings'], dim=0)
        mean_embedding = torch.mean(stacked, dim=0)
        patient_embeddings.append(mean_embedding)
        patient_labels.append(data['label'])
        patient_ids.append(subject_id)
        
    embeddings = torch.stack(patient_embeddings, dim=0)
    labels = torch.tensor(patient_labels)
    return embeddings, labels, patient_ids


def validate_per_patient(model, embeddings, labels, device):
    model.eval()
    
    embeddings = embeddings.to(device)
    with torch.no_grad():
        # on utilise la fin de la tête de classification à partir des embeddings moyennés
        outputs = model.classify_from_embedding(embeddings)
        _, predicted = torch.max(outputs, 1)
    
    predictions = predicted.cpu().numpy().tolist()
    labels_list = labels.numpy().tolist()
    accuracy = accuracy_score(labels_list, predictions)
    
    return accuracy, predictions, labels_list


def plot_embeddings_visualization(embeddings, labels, save_path, title_prefix="CNN 2D"): #les fonctions de plottings ont été générée par IA, relu puis testé par nos soins
    # Convertir en numpy si nécessaire
    if torch.is_tensor(embeddings):
        embeddings = embeddings.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Préparation des labels
    if len(labels.shape) > 1:
        y_labels = np.argmax(labels, axis=1)
    else:
        y_labels = labels
    
 
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    print("Calcul du t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(embeddings)
    
    sns.scatterplot(
        x=X_tsne[:, 0], y=X_tsne[:, 1],
        hue=y_labels,
        palette={0: 'dodgerblue', 1: 'crimson'},  # 0=sMCI (Bleu), 1=pMCI (Rouge)
        s=60, alpha=0.8, ax=axes[0]
    )
    axes[0].set_title(f't-SNE des Features du {title_prefix}', fontsize=14)
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')
    
    # Créer une légende personnalisée
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue', markersize=10, label='sMCI (MCI-4)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', markersize=10, label='pMCI (MCI-3)')
    ]
    axes[0].legend(handles=handles, loc='best')
    axes[0].grid(True, alpha=0.3)
    

    print("Calcul de UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    X_umap = reducer.fit_transform(embeddings)
    
    sns.scatterplot(
        x=X_umap[:, 0], y=X_umap[:, 1],
        hue=y_labels,
        palette={0: 'dodgerblue', 1: 'crimson'},
        s=60, alpha=0.8, ax=axes[1]
    )
    axes[1].set_title(f'UMAP des Features du {title_prefix}', fontsize=14)
    axes[1].set_xlabel('UMAP 1')
    axes[1].set_ylabel('UMAP 2')
    axes[1].legend(handles=handles, loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # Sauvegarde et Affichage
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Image sauvegardée sous : {save_path}")
    plt.close()

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    # Loss
    ax1.plot(epochs, train_losses, alpha=0.3, color='tab:blue')
    ax1.plot(epochs, val_losses, alpha=0.3, color='tab:orange')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_accs, alpha=0.3, color='tab:blue')
    ax2.plot(epochs, val_accs, alpha=0.3, color='tab:orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Historique sauvegarde: {save_path}")


def main():

    print("ENTRAINEMENT - DEBUT !")


    config = {
        'train_dir': 'adni-masked/train_dataset_2D',
        'valid_dir': 'adni-masked/valid_dataset_2D',
        'csv_file': 'adni-masked/list_standardized_tongtong_2017.csv',
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.00005,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': '2D/models',
        'num_workers': 4,
        'early_stopping_patience': 5
    }

    print(f"\nDevice: {config['device']}")
    Path(config['save_dir']).mkdir(parents=True, exist_ok=True)

    # Augmentation renforcee pour reduire l'overfitting
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    train_dataset = AlzheimerTrainDataset(config['train_dir'], config['csv_file'], train_transform)
    val_dataset = AlzheimerValidDataset(config['valid_dir'], config['csv_file'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=config['device']=='cuda')
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=config['device']=='cuda')

    model = CNN2DModel().to(config['device'])
    print(f"Modele charge sur {config['device']}")

    # Poids de classe
    train_labels = [label for _, label in train_dataset.samples]
    class_counts = [train_labels.count(i) for i in range(2)]
    class_weights = torch.FloatTensor([len(train_labels) / (2 * c) for c in class_counts]).to(config['device'])
    print(f"Poids de classe: {class_weights.tolist()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config['device'])
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, config['device'])
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}")
        print(f"Valid: loss={val_loss:.4f}, acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, Path(config['save_dir']) / 'best_model.pth')
            print(f"Meilleur modele sauvegarde (acc={val_acc:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"Pas d'amelioration depuis {epochs_without_improvement} epoch(s)")
            if epochs_without_improvement >= config['early_stopping_patience']:
                print(f"\nEarly stopping apres {epoch+1} epochs")
                break
    checkpoint = torch.load(Path(config['save_dir']) / 'best_model.pth', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extraire les embeddings par patient UNE SEULE FOIS
    # (utilisés pour l'évaluation ET la visualisation)
    embeddings, labels, patient_ids = extract_embeddings_per_patient(
        model, val_dataset, config['device'], batch_size=config['batch_size']
    )
    
    # Évaluer par patient avec les embeddings moyennés
    val_acc, val_preds, val_labels = validate_per_patient(
        model, embeddings, labels, config['device']
    )

    print(f"\nMeilleur modele (epoch {checkpoint['epoch']+1})")
    print(f"Accuracy (par patient): {val_acc:.4f} ({100*val_acc:.1f}%)")
    print(f"Nombre de patients évalués: {len(patient_ids)}")
    print(f"\nRapport de classification:")
    print(classification_report(val_labels, val_preds,
                                target_names=['Non-Alzheimer (MCI-4)', 'Alzheimer (MCI-3)']))

    plot_training_history(train_losses, val_losses, train_accs, val_accs,
                          Path(config['save_dir']) / 'training_history.png')

    # ================================================================
    # VISUALISATION t-SNE / UMAP (réutilise les embeddings déjà extraits)
    # ================================================================
    print("\n" + "=" * 60)
    print("VISUALISATION DES EMBEDDINGS (t-SNE / UMAP)")
    print("=" * 60)
    
    # Générer la visualisation avec les embeddings déjà extraits
    viz_save_path = Path(config['save_dir']) / 'visualisation_cnn_embeddings.png'
    plot_embeddings_visualization(embeddings, labels, viz_save_path, title_prefix="CNN 2D (ResNet18)")

    print("\n" + "=" * 60)
    print(f"TERMINE - Meilleure accuracy: {best_val_acc:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()