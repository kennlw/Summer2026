import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
from collections import Counter
import random
import ast

sos_val = 2
eos_val = 3
vocab_size = 61

df = pd.read_csv("num_dataset.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

inp_tensors = [torch.tensor([sos_val] + ast.literal_eval(s) + [eos_val], dtype=torch.long) for s in df['source']]
out_tensors = [torch.tensor([sos_val] + ast.literal_eval(s) + [eos_val], dtype=torch.long) for s in df['target']]

inp_padded = nn.utils.rnn.pad_sequence(inp_tensors, batch_first=True, padding_value=1)
out_padded = nn.utils.rnn.pad_sequence(out_tensors, batch_first=True, padding_value=1)

dataset = TensorDataset(inp_padded, out_padded)

#70, 15, 15 split for Train, Val, Test
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset,   batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset,  batch_size=32, shuffle=False)

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        #Expands a token to dim embed_size, based on learned embeddings
        embedding = self.dropout(self.embedding(x))

        #Output is all the hidden states, (hidden, cell) is just final context which is what we want (i.e. the context vector)
        output, (hidden, cell) = self.lstm(embedding)

        return hidden, cell
    
class Decoder(nn.Module):
    #Input and output sizes are the same here b/c both use same vocab space
    def __init__(self, input_size, embed_size, hidden_size, output_size, num_layers):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.dropout_out = nn.Dropout(0.4)

        #Fully connected layer, maps dim of lstm outputs to vocab dim
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        #Converts shape of x from (batch_size) to (1, batch_size)
        x = x.unsqueeze(1)
        #Converts shape now to (1, batch_size, embed_size)
        embedding = self.dropout(self.embedding(x))

        #Outputs = hidden here, since we're doing a sequence of length 1
        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))

        predictions = self.fc(self.dropout_out(outputs))
        predictions = predictions.squeeze(1)

        return predictions, hidden, cell
    
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio):
        batch_size = source.shape[0]
        target_len = source.shape[1]

        #Tensor of zeros with shape (batch_size, target_len, vocab_size), will gradually be filled in by the decoder
        #BN: Vocab_size != Embedding_Size
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(device)

        #Context vector
        hidden, cell = self.encoder(source)

        #Effectively the <sos> token
        # x = target[:, 0]
        x = torch.full((batch_size,), 2)

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            #Fill in the outputs variable
            outputs[:, t] = output

            #Chooses the one with the highest probability
            best_guess = output.argmax(1)

            #Using random to ensure teacher forcing is applied only a certain percentage of the time
            if random.random() < teacher_force_ratio:
                x = target[:, t]
            else:
                x = best_guess

        return outputs

#Evaluate a model (for validation and testing loss)
def evaluate(model, loader, criterion, device):
    with torch.no_grad():
        model.eval()
        epoch_loss = 0.0

        for src_batch, trg_batch in loader:
            src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
            output = model(src_batch, trg_batch, 0.0)
            output = output[:, 1:].reshape(-1, vocab_size)
            target = trg_batch[:, 1:].reshape(-1)
            loss = criterion(output, target)
            epoch_loss += loss.item()

        return epoch_loss / len(loader)
    
#Find the models accuracy on a Dataloader
def accuracy(model, loader, device):
    with torch.no_grad():
        model.eval()
        accurate = 0
        total = 0

        for src_batch, trg_batch in loader:
            src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
            output = model(src_batch, trg_batch, 0.0)

            output = output[:, 1:].reshape(-1, vocab_size)
            target = trg_batch[:, 1:].reshape(-1)

            #Creates list of booleans, False if 0, else True
            mask = target != 1
            preds = output.argmax(dim=1)

            #Ignoring padding tokens when evaluating
            accurate += (preds[mask] == target[mask]).sum().item()
            total += mask.sum().item()

        return accurate/total

def test(model, test_seq):
    with torch.no_grad():
        model.eval()
        encoded = torch.tensor([sos_val] + test_seq + [eos_val], dtype=torch.long)

        #Pad our text to the length the model was trained on
        pad_len = inp_padded.shape[1] - len(encoded)
        encoded = torch.cat([encoded, torch.zeros(pad_len, dtype=torch.long).fill_(1)])

        #Gives it batch size 1
        encoded = encoded.unsqueeze(0).to(device)

        #Context vector
        hidden, cell = model.encoder(encoded)

        predicted = []
        decoder_input = torch.tensor([sos_val], device=device)
        for step in range(len(test_seq)):
            prediction, hidden, cell = model.decoder(decoder_input, hidden, cell)
            best_guess = prediction.argmax(1).item()
            if best_guess == eos_val:
                break

            predicted.append(best_guess)
            decoder_input = torch.tensor([best_guess], device=device)
        return(predicted)

def build_context_df(model, loader, inp_padded, device):
    model.eval()
    all_sources = []
    all_targets = []
    all_contexts = []

    with torch.no_grad():
        for src_batch, trg_batch in loader:
            src_batch = src_batch.to(device)
            trg_batch = trg_batch.to(device)
            hidden, cell = model.encoder(src_batch)

            h_final = hidden[0]
            c_final = cell[0]

            # Concatenate along feature dim: (batch_size, 2 * hidden_size)
            # To undo, h_final = context_vector[:, :128], c_final = context_vector[:, 128:]
            context = torch.cat([h_final, c_final], dim=-1)

            all_sources.append(src_batch.cpu())
            all_targets.append(trg_batch.cpu())
            all_contexts.append(context.cpu())

    all_sources = torch.cat(all_sources, dim=0)   # (N, seq_len)
    all_targets = torch.cat(all_targets, dim=0)   # (N, seq_len)
    all_contexts = torch.cat(all_contexts, dim=0)  # (N, 2 * hidden_size)

    source_lists = [row.tolist() for row in all_sources]
    target_lists = [row.tolist() for row in all_targets]
    context_lists = [row.tolist() for row in all_contexts]

    context_df = pd.DataFrame({
        "source": source_lists,
        "context_vector": context_lists,
        "target": target_lists
    })

    return context_df

def strip_padding(tokens):
    while tokens and tokens[-1] == 1:
        tokens = tokens[:-1]
    return tokens

#Training hyperparameters
num_epochs = 200
learning_rate = 1e-3
encoder_embedding_size = 64
decoder_embedding_size = 64
hidden_size = 128
num_layers = 1

#Model hyperparameters
encoder_net = Encoder(vocab_size, encoder_embedding_size, hidden_size, num_layers).to(device)
decoder_net = Decoder(vocab_size, decoder_embedding_size, hidden_size, vocab_size, num_layers).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
#Ignore <pad>
criterion = nn.CrossEntropyLoss(ignore_index=1)
#Combat overfitting
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7
)

test_str = [29, 5, 38, 5, 25, 46, 32]
tf_ratio = 1
train_losses = []
val_losses = []
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} / {num_epochs}")
    model.train()

    epoch_loss = 0.0
    with torch.set_grad_enabled(True):
      for src_batch, trg_batch in train_loader:
          src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
          optimizer.zero_grad()

          output = model(src_batch, trg_batch, max(0.0, tf_ratio))

          #Skip <sos>, then flatten the batch size and seq length dimensions into 1
          output = output[:, 1:].reshape(-1, vocab_size)
          target = trg_batch[:, 1:].reshape(-1)

          loss = criterion(output, target)
          loss.backward()

          #Norm=magnitude, clipping gradient to ensure they don't explode
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
          optimizer.step()
          epoch_loss += loss.item()

    print(f"{test_str} => {test(model, test_str)}")
    train_loss = epoch_loss/len(train_loader)
    print(f"Train Loss: {train_loss}")
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss}")
    tf_ratio -= 0.01
    scheduler.step(val_loss)

print("----------------Final----------------")
print(f"Train Loss: {evaluate(model, train_loader, criterion, device)}")
print(f"Train Acc: {accuracy(model, train_loader, device)}")
print(f"Val Loss: {evaluate(model, val_loader, criterion, device)}")
print(f"Val Acc: {accuracy(model, val_loader, device)}")
print(f"Test Loss: {evaluate(model, test_loader, criterion, device)}")
print(f"Test Acc: {accuracy(model, test_loader, device)}")

# torch.save(model.state_dict(), "num_identity_model.pt")
model.load_state_dict(torch.load("num_reversal_model.pt", weights_only=True, map_location=torch.device('cpu')))

full_loader = DataLoader(dataset, batch_size=32, shuffle=False)
context_df = build_context_df(model, full_loader, inp_padded, device)

context_df["source"] = context_df["source"].apply(strip_padding)
context_df["target"] = context_df["target"].apply(strip_padding)

# context_df.to_csv("identity_data.csv")
context_df.to_csv("reversal_data.csv")