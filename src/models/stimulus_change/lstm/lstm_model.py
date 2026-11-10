import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, hidden_size // 2)
        self.output = nn.Linear(hidden_size // 2, output_size)

    def forward_train(self, seq, labels, sampling_prob=0.0):
        device = seq.device
        num_classes = self.output.out_features

        outputs = []
        labels_out = []


        prev_one_hot = torch.zeros(num_classes, device=device)

        h = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=device)
        c = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size, device=device)

        for t in range(len(seq)):
            xt_full = torch.cat([seq[t], prev_one_hot], dim=0)
            xt_full = xt_full.unsqueeze(0).unsqueeze(0)

            lstm_out, (h, c) = self.lstm(xt_full, (h, c))
            hidden = self.fc(lstm_out)
            logits = self.output(hidden)

            outputs.append(logits[0, 0])
            labels_out.append(labels[t])

            if torch.rand(1).item() < sampling_prob:
                pred_class = torch.argmax(logits[0, 0])
            else:
                pred_class = labels[t]

            prev_one_hot = torch.zeros(num_classes, device=device)
            prev_one_hot[pred_class] = 1

        return torch.stack(outputs), torch.tensor(labels_out, device=device)

    def forward_test(self, x):
        num_classes = self.output.out_features
        seq_len = x.size(0)

        prev_decision_one_hot = torch.zeros(num_classes)

        h = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size)
        c = torch.zeros(self.lstm.num_layers, 1, self.lstm.hidden_size)

        predictions = []
        last_logits = None

        for t in range(seq_len):
            xt_full = torch.cat([x[t], prev_decision_one_hot], dim=0)  # shape (input_size,)
            xt_full = xt_full.unsqueeze(0).unsqueeze(0)

            lstm_out, (h, c) = self.lstm(xt_full, (h, c))

            hidden = self.fc(lstm_out)
            logits = self.output(hidden)

            pred = torch.argmax(logits, dim=2)[0, 0].item()
            predictions.append(pred)

            prev_decision_one_hot = torch.zeros(num_classes)
            prev_decision_one_hot[pred] = 1

            last_logits = logits[0, 0]

        return torch.tensor(predictions), last_logits