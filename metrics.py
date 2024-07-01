import pandas as pd
import torch
from tqdm.notebook import tqdm


def calculate_gap(model, data_loader, train_labels, device):
    """
    Calculate the Global Average Precision (GAP) on the validation dataset.

    Args:
    - model (torch.nn.Module): The trained PyTorch model.
    - data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - train_labels (array): Unique indexes of landmarks of the training dataset
    - device (torch.device): The device to run the model on (CPU or CUDA).

    Returns:
    - float: GAP score.
    - df: Sorted by confidence dataframe with predictions
    """
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    all_predictions = []
    all_labels = []
    all_confidences = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Evaluating model'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            populate_confidence(all_confidences, all_labels, all_predictions, labels, outputs)

    return gap_metric(all_confidences, all_labels, all_predictions, train_labels)


def gap_metric(all_confidences, all_labels, all_predictions, train_labels):
    # Create a DataFrame to store predictions and labels
    df = pd.DataFrame({
        'label_index': all_labels,
        'prediction': all_predictions,
        'confidence': all_confidences
    })
    # Sort by confidence in descending order
    df_sorted = df.sort_values(by='confidence', ascending=False).reset_index(drop=False)
    # Initialize variables
    correct_predictions = 0
    total_relevant = 0
    gap = 0.0
    for i, row in df_sorted.iterrows():
        if row['label_index'] in train_labels:
            total_relevant += 1
            if row['label_index'] == row['prediction']:
                correct_predictions += 1
                precision_at_i = correct_predictions / total_relevant
                gap += precision_at_i
    gap /= total_relevant
    return gap, df_sorted


def populate_confidence(all_confidences, all_labels, all_predictions, labels, outputs):
    confidences, predictions = torch.max(torch.softmax(outputs, dim=1), dim=1)
    all_predictions.extend(predictions.cpu().detach().numpy())
    all_labels.extend(labels.cpu().detach().numpy())
    all_confidences.extend(confidences.cpu().detach().numpy())


def accuracy(model, data_loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = 100 * total_correct / total_samples
    return accuracy
