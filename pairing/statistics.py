import torchmetrics
import torch

from pairing.data import PairData
import pairing.data

import scipy


def get_y(data):
    ys = []
    for d in data:
        ys.append(d.y)
    return torch.stack(ys, dim=0)

# Baseline auroc using mean of labels is 0.5
def baseline(average=True):
    auroc = torchmetrics.classification.MultilabelAUROC(
        pairing.data.Dataset.num_classes(),average=None)

    train_data = pairing.data.Dataset(is_train=True)
    train_y = get_y(train_data)
    
    test_data = pairing.data.Dataset(is_train=False)
    test_y = get_y(test_data)

    pred = train_y.mean(dim=0).unsqueeze(0)
    pred = pred.expand(len(test_y), -1)
    score = auroc(pred, test_y.int())
    print(score)
    exit()
    print(f"Baseline auroc using mean of labels is {score}")


def distribution():
    train_data = pairing.data.Dataset(is_train=True)
    train_y = get_y(train_data)

    test_data = pairing.data.Dataset(is_train=False)
    test_y = get_y(test_data)

    all_y = torch.cat([train_y, test_y])

    train_diff = all_y.mean(dim=0) - train_y.mean(dim=0)
    test_diff = all_y.mean(dim=0) - test_y.mean(dim=0)
    train_test = train_y.mean(dim=0) - test_y.mean(dim=0)

    # Could use a pairwise euclidian distance but that would be very expensive.
    print(
        f"Train data varies from dataset by a rating of {train_diff.std():.6f}."
    )
    print(
        f"Test data varies from dataset by a rating of {test_diff.std():.6f}.")
    print(
        f"Train data varies from test data by a rating of {train_test.std():.6f}."
    )


if __name__ == "__main__":
    baseline()
    # distribution()
