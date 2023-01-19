import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import wandb

import click

from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(lr)

    wandb.init()

    model = MyAwesomeModel()
    train_data, _ = mnist()
    images, labels = train_data
    train_set = TensorDataset(images, labels)

    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    num_epochs = 10
    loss_history = []

    wandb.watch(model, log_freq=100)

    model.train()

    for e in range(num_epochs):
        loss = 0
        print(e + 1)
        for input, target in train_loader:
            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, target.long())
            loss.backward()
            optimizer.step()
            loss = +loss.detach().cpu().item()
        wandb.log({"loss": loss})
        print(loss)
        loss_history.append(loss)

    fig = plt.figure()
    plt.plot(loss_history)
    figure_path = "figures/train_loss.jpeg"
    plt.savefig(figure_path)

    wandb.log({"fig": fig})

    model_path = "models/trained_model.pt"
    torch.save(model, model_path)


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = torch.load(model_checkpoint).eval()

    _, test_data = mnist()
    images, labels = test_data
    test_set = TensorDataset(images, labels)

    test_loader = DataLoader(test_set)

    correct = 0

    for input, target in test_loader:
        output = model(input)
        correct = correct + torch.sum(torch.argmax(output) == target).item()

    accuracy = correct / len(test_set)
    print(accuracy)


cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
