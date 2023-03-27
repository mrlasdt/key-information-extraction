import torch
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from config import config as cfg
from sty import fg, bg, ef, rs
from terminaltables import AsciiTable


class Trainer:
    def __init__(self, model, optimizer, kie_labels, save_dir, n_epoches):
        self.model = model
        self.optimizer = optimizer
        self.kie_labels = kie_labels
        self.save_dir = save_dir
        self.n_epoches = n_epoches

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        total, correct = 0, 0
        for batch in tqdm(dataloader):
            # forward pass
            outputs = self.model(**batch)
            loss = outputs.loss

            running_loss += loss.item()
            predictions = outputs.logits.argmax(dim=2)

            valid_samples = (batch['labels'] != -100)
            predictions = predictions[valid_samples]
            batch_labels = batch['labels'][valid_samples]
            correct += (predictions == batch_labels).float().sum()
            total += predictions.numel()

            # backward pass to get the gradients
            loss.backward()

            # update
            self.optimizer.step()
            self.optimizer.zero_grad()

        print("Loss:", running_loss / batch["input_ids"].shape[0])
        accuracy = 100 * correct / total
        print("Training accuracy:", accuracy.item())

    def val_one_epoch(self, dataloader):
        self.model.eval()
        total, correct = 0, 0
        preds, truths = [], []
        for batch in tqdm(dataloader):
            with torch.no_grad():
                outputs = self.model(**batch)
                predictions = outputs.logits.argmax(dim=2)
                valid_samples = (batch['labels'] != -100)
                predictions = predictions[valid_samples]
                batch_labels = batch['labels'][valid_samples]
                # encoded_sequence = batch['input_ids'].detach().cpu().numpy().tolist()
                # decoded_sequence = [tokenizer.decode(i) for i in encoded_sequence]
                # decoded_sequence = [list(i) for i in decoded_sequence]
                # print("len decode",len(decoded_sequence[0]))
                # print(len(decoded_sequence))
                # decoded_str=[]
                # print("new_batch")
                # for l in decoded_sequence:
                #     x =""
                #     # print(l)
                #     for char in l:
                #         x+= char
                #     x = x.split()
                #     x.pop(0)
                #     print(x)
                #     print(len(x), x[0], x[-1])
                #     decoded_str.extend(x)

                # print('fhakjsfdh')
                # tru = batch_labels.detach().cpu().numpy()
                # print(tru.shape)
                # print([KIE_LABELS[i] for i in tru])

                preds.extend(predictions.detach().cpu().numpy().tolist())
                truths.extend(batch_labels.detach().cpu().numpy().tolist())
                correct += (predictions == batch_labels).float().sum()
                total += predictions.numel()

        accuracy = 100 * correct / total
        p, r, f1, support = precision_recall_fscore_support(truths, preds)
        # print("shapeeee", p.shape)
        table_data = [["Class", "P", "R", "F1", "#samples"]]
        for c in range(len(self.kie_labels)):
            if c < p.shape[0]:
                table_data.append([self.kie_labels[c], p[c], r[c], f1[c], support[c]])
            continue
        table = AsciiTable(table_data)
        print(table.table)
        print(
            "Validation accuracy:", accuracy.item(),
            "- #samples:", total, "- #corrects:", correct
        )
        return accuracy

    def update_metric_and_save_model(self, best_acc, acc):
        if acc > best_acc:
            print(f"{fg.green}Accuracy updated from {best_acc} to {acc}{fg.rs}")
            best_acc = acc
            print("save new best model")
            self.model.save_pretrained(self.save_dir)
        print(f"{fg.blue} Current best accuracy: {best_acc}{fg.rs}")
        return best_acc

    def train(self, train_dataloader, val_dataloader):
        r"""Train LayoutLM model"""

        best_acc = 0.0
        for epoch in range(self.n_epoches):
            print("Epoch:", epoch)
            self.train_one_epoch(train_dataloader)
            acc = self.val_one_epoch(val_dataloader)
            best_acc = self.update_metric_and_save_model(best_acc, acc)


def load_trainer(model, optimizer, kie_labels, save_dir, n_epoches):
    return Trainer(model, optimizer, kie_labels, save_dir, n_epoches)
