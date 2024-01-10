from typing import cast
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

        batches_losses = list[float]()
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
            batches_losses.append(loss)

            if (batch_num + 1) % print_interval == 0:
                mean_batch_loss = np.average(np.array(batches_losses))
                batches_losses.clear()
                print(f"Batch {batch_num+1:>5} average loss: {mean_batch_loss:>7f}")

            if (batch_num + 1) % save_interval == 0:
                # Save the model.
                during_epoch_filename = f"1_Epoch_{epoch}_0_During_{batch_num+1}.model"
                during_epoch_filepath = os.path.join(save_dir, during_epoch_filename)
                torch.save(model.state_dict(), during_epoch_filepath)
        after = time.time()
        num_batches = batch_num

        # Save the model after the epoch.
        after_epoch_filename = f"1_Epoch_{epoch}_1_After.model"
        after_epoch_filepath = os.path.join(save_dir, after_epoch_filename)
        torch.save(model.state_dict(), after_epoch_filepath)

        mean_loss = np.average(np.array(losses))
        print(f"Average loss: {mean_loss:>7f} [{(epoch+1):>5}]/{num_epochs:>5}")
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
    # dataset = datasets.OancBatchedSnippetsDataset(
    #     snippets_dir=r"Datasets\Oanc_Snippets_Len11"
    # )
    dataset = datasets.OancBatchedDataset(dataset_dir=r"Datasets\Oanc_Len11_Tensors")

    # model = models.PlainNn().to(device)
    # model = models.ComplexNn().to(device)
    # model = models.PathComplexNn().to(device)
    # model = models.SimpleLetterModel().to(device)
    model = models.ConvLetterModel().to(device)
    model.train()
    model.init_model()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=1e-2, momentum=0.99
    )  # , momentum=0.3)

    num_epochs = 100000
    print_interval = 500
    save_interval = 10000
    save_dir = r"Model_Saves\Conv_5"

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


# Save the dataset as .pt files for faster(?) loading.
def save_dataset_as_tensors():
    dataset = datasets.OancBatchedSnippetsDataset(
        snippets_dir=r"Datasets\Oanc_Snippets_Len11"
    )

    out_dir = r"Datasets\Oanc_Len11_Tensors"
    batch_input_template = "batch_{}_input.pt"
    batch_label_template = "batch_{}_label.pt"

    dataloader = DataLoader(
        dataset=dataset, batch_size=None, shuffle=True, num_workers=14
    )
    for batch_num, (x, y) in enumerate(dataloader):
        print(f"Saving batch {batch_num}.")
        batch_input_filename = batch_input_template.format(batch_num)
        batch_input_filepath = os.path.join(out_dir, batch_input_filename)
        x = x.detach().clone()
        torch.save(x, batch_input_filepath)

        batch_label_filename = batch_label_template.format(batch_num)
        batch_label_filepath = os.path.join(out_dir, batch_label_filename)
        y = y.detach().clone()
        torch.save(y, batch_label_filepath)


def split_up_batches():
    in_dir = r"Datasets\Oanc_Len11_Tensors_3"
    out_dir = r"Datasets\Oanc_Len11_Tensors"

    num_splits_per_batch = 128

    num_in_batches = 632
    # num_out_batches = 632 * num_splits_per_batch
    out_batch_num = 0
    for in_batch_num in range(num_in_batches):
        print(f"Batch {in_batch_num}.")
        in_batch_inputs_filename = f"batch_{in_batch_num}_input.pt"
        in_batch_inputs_filepath = os.path.join(in_dir, in_batch_inputs_filename)
        in_batch_inputs = cast(
            torch.Tensor, torch.load(in_batch_inputs_filepath).detach().clone()
        )

        in_batch_labels_filename = f"batch_{in_batch_num}_label.pt"
        in_batch_labels_filepath = os.path.join(in_dir, in_batch_labels_filename)
        in_batch_labels = cast(
            torch.Tensor, torch.load(in_batch_labels_filepath).detach().clone()
        )

        in_num_samples = in_batch_inputs.shape[0]
        if in_num_samples % num_splits_per_batch != 0:
            print(f"Batch {in_batch_num} not evenly divided: {in_num_samples} samples.")
        out_num_samples = in_num_samples // num_splits_per_batch

        # Split this batch into multiple batches.
        for split_batch_num in range(num_splits_per_batch):
            # Split up the inputs.
            out_batch_inputs = in_batch_inputs[
                split_batch_num
                * out_num_samples : (split_batch_num + 1)
                * out_num_samples,
                :,
                :,
            ]
            out_batch_inputs = out_batch_inputs.detach().clone()

            # Save the inputs.
            out_batch_inputs_filename = f"batch_{out_batch_num}_input.pt"
            out_batch_inputs_filepath = os.path.join(out_dir, out_batch_inputs_filename)
            torch.save(out_batch_inputs, out_batch_inputs_filepath)

            # Split up the outputs.
            out_batch_labels = in_batch_labels[
                split_batch_num
                * out_num_samples : (split_batch_num + 1)
                * out_num_samples,
                :,
            ]
            out_batch_labels = out_batch_labels.detach().clone()

            # Save the outputs.
            out_batch_labels_filename = f"batch_{out_batch_num}_label.pt"
            out_batch_labels_filepath = os.path.join(out_dir, out_batch_labels_filename)
            torch.save(out_batch_labels, out_batch_labels_filepath)

            out_batch_num += 1


def get_num(filename: str) -> int:
    return int(filename.split("_")[1])


def check_files():
    in_dir = r"Datasets\Oanc_Len11_Tensors"
    out_file = "data_files.txt"
    names = list[str]()
    for filename in os.listdir(in_dir):
        # print(filename)
        # file.write(f"{filename}\n")
        names.append(filename)
    names.sort(key=lambda n: get_num(n))
    with open(out_file, "w") as file:
        for name in names:
            file.write(f"{name}\n")


def main():
    do_train()
    # save_dataset_as_tensors()
    # split_up_batches()
    # check_files()


if __name__ == "__main__":
    main()
