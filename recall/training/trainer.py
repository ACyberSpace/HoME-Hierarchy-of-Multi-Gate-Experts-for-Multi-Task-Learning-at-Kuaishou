import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from recall.models.swing import Swing
from recall.models.item2vec import Item2Vec
from recall.models.dssm import DSSM
from recall.models.mind import MIND
from recall.models.sdm import SDM
from recall.models.youtubednn import YouTubeDNN


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


def _in_batch_softmax_loss(user_repr, item_repr, temperature=0.05):
    user_repr = F.normalize(user_repr, dim=1)
    item_repr = F.normalize(item_repr, dim=1)
    logits = torch.matmul(user_repr, item_repr.t()) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)


def train_dssm(model, dataloader, epochs=5, lr=0.001, temperature=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            user_features = {k: v.long().to(device) for k, v in batch["user_features"].items()}
            pos_item_features = {k: v.long().to(device) for k, v in batch["pos_item_features"].items()}

            user_repr = model.get_user_repr(user_features)
            item_repr = model.get_item_repr(pos_item_features)
            loss = _in_batch_softmax_loss(user_repr, item_repr, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


def train_youtubednn(model, dataloader, epochs=5, lr=0.001, temperature=0.05):
    train_dssm(model, dataloader, epochs=epochs, lr=lr, temperature=temperature)


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


def train_sdm(model, dataloader, epochs=5, lr=0.001, temperature=0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            user_features = {k: v.long().to(device) for k, v in batch["user_features"].items()}
            item_features = {k: v.long().to(device) for k, v in batch["item_features"].items()}

            user_repr = model.get_user_repr(user_features)
            item_repr = model.get_item_repr(item_features)
            loss = _in_batch_softmax_loss(user_repr, item_repr, temperature)
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
        train_dssm(
            model,
            data_loader,
            epochs=config.get("epochs", 5),
            lr=config.get("lr", 0.001),
            temperature=config.get("softmax_temperature", 0.05),
        )
    elif model_type == 'youtubednn':
        train_youtubednn(
            model,
            data_loader,
            epochs=config.get("epochs", 5),
            lr=config.get("lr", 0.001),
            temperature=config.get("softmax_temperature", 0.05),
        )
    elif model_type == 'mind':
        train_mind(model, data_loader, epochs=config.get("epochs", 5), lr=config.get("lr", 0.001))
    elif model_type == 'sdm':
        train_sdm(
            model,
            data_loader,
            epochs=config.get("epochs", 5),
            lr=config.get("lr", 0.001),
            temperature=config.get("softmax_temperature", 0.05),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
