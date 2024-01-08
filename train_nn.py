import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import models
import datasets


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_data():
    pass


# Train the NN for one epoch.
def train(
    # input_data,
    dataset,
    model,
    loss_fn,
    optimizer,
    num_epochs: int,
    print_interval: int,
    save_interval: int,
    save_dir: str,
):
    # before = time.time()
    before_train_filename = "0_Before_Training.model"
    before_train_filepath = os.path.join(save_dir, before_train_filename)

    # Save the model as it is before training.
    torch.save(model.state_dict(), before_train_filepath)

    model.train()
    # X, y = input_data.to(device), output_data.to(device)
    dataloader = DataLoader(
        dataset=dataset, batch_size=None, shuffle=True, num_workers=14
    )

    # ep_w = len(str(num_epochs))

    # if batch_num % 100 == 0:
    for epoch in range(num_epochs):
        losses = list[float]()
        print(f"Epoch {epoch+1} out of {num_epochs}:")
        batch_num = 0
        before = time.time()
        for batch_num, (x, y) in enumerate(dataloader):
            # print(f"x.shape: {x.shape}")
            # print(f"y.shape: {y.shape}")
            # print("")
            x = x.to(device)
            y = y.to(device)

            # Compute prediction error
            pred = model(x)
            loss = loss_fn(pred, y)

            # Backpropagate the loss.
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss = loss.item()  # , (batch_num + 1) * len(X)
            losses.append(loss)

            if (batch_num + 1) % print_interval == 0:
                print(f"Batch {batch_num:>5} loss: {loss:>7f}")

            if (batch_num + 1) % save_interval == 0:
                # Save the model.
                during_epoch_filename = f"1_Epoch_{epoch}_0_During_{batch_num}.model"
                during_epoch_filepath = os.path.join(save_dir, during_epoch_filename)
                torch.save(model.state_dict(), during_epoch_filepath)
        after = time.time()
        num_batches = batch_num

        # Save the model after the epoch.
        after_epoch_filename = f"1_Epoch_{epoch}_1_After.model"
        after_epoch_filepath = os.path.join(save_dir, after_epoch_filename)
        torch.save(model.state_dict(), after_epoch_filepath)

        mean_loss = np.average(np.array(losses))
        print(f"loss: {mean_loss:>7f} [{(epoch+1):>5}]/{num_epochs:>5}")
        print(f"Took {after-before:.02f} seconds for {num_batches} batches,")
        print(f"for {(after-before)/num_batches:.04f} seconds per batch.")
        print("")
    
    final_filename = f"2_Final.model"
    final_filepath = os.path.join(save_dir, final_filename)
    torch.save(model.state_dict(), final_filepath)

    # after = time.time()
    # print(f"Took {(after-before):>1f} seconds.")


def do_train():
    # input_data, output_data = get_data()
    # dataset = datasets.OancSnippetsDataset(snippets_dir=r"Datasets\Oanc_Snippets_Len5")
    dataset = datasets.OancBatchedSnippetsDataset(
        snippets_dir=r"Datasets\Oanc_Snippets_Len5"
    )

    # model = models.PlainNn().to(device)
    # model = models.ComplexNn().to(device)
    # model = models.PathComplexNn().to(device)
    model = models.SimpleLetterModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-1, momentum=0.9
    )  # , momentum=0.3)

    num_epochs = 1
    print_interval = 1
    save_interval = 25
    save_dir = r"Model_Saves\Fully_Connected"

    # for _ in range(num_epochs):
    train(
        dataset=dataset,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=num_epochs,
        print_interval=print_interval,
        save_interval=save_interval,
        save_dir=save_dir,
    )


def main():
    do_train()


if __name__ == "__main__":
    main()
