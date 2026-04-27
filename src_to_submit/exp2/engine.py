import os
import time

import torch

from utils import progress_bar, sparse_selection


def train_one_epoch(epoch, net, trainloader, optimizer, criterion, device):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        sparse_selection()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(
            batch_idx,
            len(trainloader),
            "Loss: %.3f | Acc: %.3f (%d/%d)"
            % (train_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
        )

    train_acc = 100.0 * correct / total if total > 0 else 0.0
    return train_loss / len(trainloader), train_acc


def evaluate(epoch, net, testloader, optimizer, criterion, scheduler, args, best_acc, device):
    net.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(
                batch_idx,
                len(testloader),
                "Loss: %.3f | Acc: %.3f (%d/%d)"
                % (test_loss / (batch_idx + 1), 100.0 * correct / total, correct, total),
            )

    test_loss = test_loss / len(testloader)
    if not args.cos:
        scheduler.step(test_loss)

    acc = 100.0 * correct / total
    if acc > best_acc:
        print("saving..")
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")
        torch.save(state, f"./checkpoint/{args.net}-{args.patch}-ckpt.t7")
        best_acc = acc

    os.makedirs("log", exist_ok=True)
    content = (
        f"{time.ctime()} Epoch {epoch}, lr: {optimizer.param_groups[0]['lr']:.7f}, "
        f"val loss: {test_loss:.7f}, val acc: {acc:.3f}"
    )
    print(content)
    with open(f"log/log_{args.net}_patch{args.patch}.txt", "a", encoding="utf-8") as appender:
        appender.write(content + "\n")

    return test_loss, acc, best_acc
