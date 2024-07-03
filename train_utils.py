import torch
from torch.utils.data import Dataset, DataLoader
import  metrics
from tqdm.notebook import tqdm
import PIL.Image
import matplotlib.pyplot as plt

def get_device():
    print(torch.version.cuda)
    print("cuda available: " + str(torch.cuda.is_available()))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f"device: {device}")
    return device


class CustomCountryDataset(Dataset):
    def __init__(self, photo_label_df, path, transform):
        self.photo_label_df = photo_label_df
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.photo_label_df)

    def __getitem__(self, index):
        row = self.photo_label_df.iloc[index]
        image_id = row['id']
        landmark_id_index = row['label_index']  # Use the label index
        file = image_id + '.jpg'
        subpath = '/'.join([char for char in image_id[0:3]])

        image = PIL.Image.open(self.path + '/' + subpath + '/' + file)
        X = self.transform(image)
        y = torch.tensor(landmark_id_index, dtype=torch.long)

        return X, y


def get_dataloaders(train_df, val_df, base_path, transform, batch_size=64):
    train_dataset = CustomCountryDataset(
        photo_label_df=train_df,
        path=base_path + '/train',
        transform=transform,
    )

    val_dataset = CustomCountryDataset(
        photo_label_df=val_df,
        path=base_path + '/train',
        transform=transform,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_dataloader, val_dataloader


def train(model,
          train_data_loader,
          validation_data_loader,
          optimizer,
          criterion,
          epochs,
          train_class_labels,
          checkpoint_path,
          device):
    model.to(device)

    results = {
        "train_loss": [],
        "train_gap": [],
        "val_loss": [],
        "val_gap": []
    }

    best_val_gap = 0.0

    for epoch in range(epochs):

        model.train()
        running_loss = 0.0

        train_predictions = []
        train_labels = []
        train_confidences = []

        for images, labels in tqdm(train_data_loader, desc='Training loop'):
            # Print labels to debug
            # print(f"Labels before moving to device: {labels}")
            # print(f"Labels dtype: {labels.dtype}, Labels min: {labels.min()}, Labels max: {labels.max()}")

            # Ensure labels are within the valid range
            # assert labels.min() >= 0 and labels.max() < 339, "Label out of range"

            # Print images shape and check for NaN or Inf values
            # print(f"Images shape: {images.shape}")
            assert not torch.isnan(images).any(), "Image tensor contains NaN"
            assert not torch.isinf(images).any(), "Image tensor contains Inf"

            # Ensure images are of type float
            images = images.float()
            # Move inputs and labels to the device
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            metrics.populate_confidence(train_confidences, train_labels, train_predictions, labels, outputs)

        train_loss = running_loss / len(train_data_loader.dataset)
        train_gap, _ = metrics.gap_metric(train_confidences, train_labels, train_predictions, train_class_labels)

        results["train_loss"].append(train_loss)
        results["train_gap"].append(train_gap)

        # Validation phase
        model.eval()
        running_loss = 0.0

        val_predictions = []
        val_labels = []
        val_confidences = []

        with torch.no_grad():
            for images, labels in tqdm(validation_data_loader, desc='Validation loop'):
                # Move inputs and labels to the device
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)

                metrics.populate_confidence(val_confidences, val_labels, val_predictions, labels, outputs)
        valid_loss = running_loss / len(validation_data_loader.dataset)
        valid_gap, _ = metrics.gap_metric(val_confidences, val_labels, val_predictions, train_class_labels)

        results["val_loss"].append(valid_loss)
        results["val_gap"].append(valid_gap)

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_GAP: {train_gap:.4f} | "
            f"val_loss: {valid_loss:.4f} | "
            f"val_GAP: {valid_gap:.4f} "
        )

        # Checkpoint saving logic
        if valid_gap > best_val_gap:
            best_val_gap = valid_gap
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': valid_loss,
                        'gap': valid_gap},
                       checkpoint_path)

            print(f"Checkpoint saved at epoch {epoch + 1} with validation GAP {valid_gap}")

    return results

def plot_loss_curves(results):

    loss = results["train_loss"]
    test_loss = results["val_loss"]

    accuracy = results["train_gap"]
    test_accuracy = results["val_gap"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_GAP")
    plt.plot(epochs, test_accuracy, label="val_GAP")
    plt.title("GAP")
    plt.xlabel("Epochs")
    plt.legend()