import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from recall.models.swing import Swing
from recall.models.item2vec import Item2Vec
from recall.models.dssm import DSSM
from recall.models.mind import MIND
from recall.models.sdm import SDM


def train_swing(model, data_loader):
    data_loader.load_data()
    user_item_pairs = data_loader.get_user_item_pairs()
    model.fit(user_item_pairs)
    print("Swing训练完成")


def train_item2vec(model, data_loader):
    data_loader.load_data()
    sequences = data_loader.generate_samples()
    model.fit(sequences)
    print("Item2Vec训练完成")


def train_dssm(model, dataloader, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            user_features = {k: v.long().to(device) for k, v in batch["user_features"].items()}
            pos_item_features = {k: v.long().to(device) for k, v in batch["pos_item_features"].items()}
            neg_item_features_list = [{k: v.long().to(device) for k, v in neg.items()} for neg in batch["neg_item_features_list"]]

            pos_score = model(user_features, pos_item_features)
            pos_label = torch.ones_like(pos_score)

            neg_scores = []
            for neg_features in neg_item_features_list:
                neg_score = model(user_features, neg_features)
                neg_scores.append(neg_score)
            neg_scores = torch.stack(neg_scores, dim=1)
            neg_label = torch.zeros_like(neg_scores)

            scores = torch.cat([pos_score.unsqueeze(1), neg_scores], dim=1)
            labels = torch.cat([pos_label.unsqueeze(1), neg_label], dim=1)

            loss = criterion(torch.sigmoid(scores), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


def train_mind(model, dataloader, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            user_features = {k: v.long().to(device) for k, v in batch["user_features"].items()}
            item_features = {k: v.long().to(device) for k, v in batch["item_features"].items()}
            labels = batch["label"].float().to(device)

            scores = model(user_features, item_features)
            loss = criterion(torch.sigmoid(scores), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


def train_sdm(model, dataloader, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            user_features = {k: v.long().to(device) for k, v in batch["user_features"].items()}
            item_features = {k: v.long().to(device) for k, v in batch["item_features"].items()}
            labels = batch["label"].float().to(device)

            scores = model(user_features, item_features)
            loss = criterion(torch.sigmoid(scores), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


def train_model(model_type: str, model, data_loader, config: dict):
    if model_type == 'swing':
        train_swing(model, data_loader)
    elif model_type == 'item2vec':
        train_item2vec(model, data_loader)
    elif model_type == 'dssm':
        train_dssm(model, data_loader, epochs=config.get("epochs", 5), lr=config.get("lr", 0.001))
    elif model_type == 'mind':
        train_mind(model, data_loader, epochs=config.get("epochs", 5), lr=config.get("lr", 0.001))
    elif model_type == 'sdm':
        train_sdm(model, data_loader, epochs=config.get("epochs", 5), lr=config.get("lr", 0.001))
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
