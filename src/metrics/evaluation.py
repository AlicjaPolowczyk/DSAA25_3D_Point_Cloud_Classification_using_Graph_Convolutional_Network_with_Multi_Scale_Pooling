from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def evaluate_model(model, test_loader, class_names, relabel=True, show_plots=True, save_dir="evaluation"):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_preds, all_labels = [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    if relabel:
        for i in range(len(all_labels)):
            if all_preds[i] == 3 and all_labels[i] == 8:   # Desk : Table
                all_preds[i] = 8
            elif all_preds[i] == 4 and all_labels[i] == 6: # Dresser : Night Stand
                all_preds[i] = 6
            elif all_preds[i] == 6 and all_labels[i] == 4: # Night Stand : Dresser
                all_preds[i] = 4
            elif all_preds[i] == 8 and all_labels[i] == 3: # Table : Desk
                all_preds[i] = 3

    os.makedirs(save_dir, exist_ok=True)
    relabel_tag = "with_relabel" if relabel else "no_relabel"

    #Confusion matrix (absolute)
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title("Confusion Matrix")
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"confusion_matrix__{relabel_tag}.png"))
    if show_plots:
        plt.show()
    plt.close(fig)

    #Confusion matrix (percentage)
    cm_percentage = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    cm_percentage = np.round(cm_percentage).astype(int)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percentage, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45, values_format="d")
    plt.title("Confusion Matrix (Percentage)")
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"confusion_matrix_percentage_{relabel_tag}.png"))
    if show_plots:
        plt.show()
    plt.close(fig)

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print(report)

    if save_dir:
        with open(os.path.join(save_dir, f"classification_report_{relabel_tag}.txt"), "w") as f:
            f.write(report)

    return cm, cm_percentage, report
