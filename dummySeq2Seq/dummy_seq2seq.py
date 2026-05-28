import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pandas as pd
from collections import Counter
import random

df = pd.read_csv("new_dataset.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Get all words in the dataset in one list
all_tokens = []
for sentence in (df['source'].tolist()):
    for word in sentence.lower().split():
        all_tokens.append(word)

#Assign value to each word based on how common it is
vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
for token, _ in Counter(all_tokens).most_common():
    vocab[token] = len(vocab)

#Return list of token values for a sentence, all unknowns are given the <unk> token (3)
def encode(sentence):
    return [vocab.get(tok, vocab['<unk>']) for tok in sentence.lower().split()]

#Encode with <sos> and <eos> tags
def encode_with_special(sentence):
    tokens = [vocab['<sos>']] + encode(sentence) + [vocab['<eos>']]
    return torch.tensor(tokens, dtype=torch.long)

inp_tensors = [encode_with_special(s) for s in df['source']]
out_tensors = [encode_with_special(s) for s in df['target']]
# out_tensors = [encode_with_special(s) for s in df['source']]

#Pad to match length of longest sequence
inp_padded = nn.utils.rnn.pad_sequence(inp_tensors, batch_first=True, padding_value=0)
out_padded = nn.utils.rnn.pad_sequence(out_tensors, batch_first=True, padding_value=0)

dataset = TensorDataset(inp_padded, out_padded)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

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

    def forward(self, x):
        #Expands a token to dim embed_size, based on learned embeddings
        embedding = self.embedding(x)

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

        #Fully connected layer, maps dim of lstm outputs to vocab dim
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        #Converts shape of x from (batch_size) to (1, batch_size)
        x = x.unsqueeze(1)
        #Converts shape now to (1, batch_size, embed_size)
        embedding = self.embedding(x)

        #Outputs = hidden here, since we're doing a sequence of length 1
        outputs, (hidden, cell) = self.lstm(embedding, (hidden, cell))
        
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(1)

        return predictions, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        vocab_size = len(vocab)

        #Tensor of zeros with shape (batch_size, target_len, vocab_size), will gradually be filled in by the decoder
        #BN: Vocab_size != Embedding_Size
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(device)

        #Context vector
        hidden, cell = self.encoder(source)

        #Effectively the <sos> token
        x = target[:, 0]

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
            output = output[:, 1:].reshape(-1, len(vocab))
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
            output = output[:, 1:].reshape(-1, len(vocab))
            target = trg_batch[:, 1:].reshape(-1)

            #Creates list of booleans, False if 0, else True
            mask = target != 0
            preds = output.argmax(dim=1)

            #Ignoring padding tokens when evaluating
            accurate += (preds[mask] == target[mask]).sum().item()
            total += mask.sum().item()

        return accurate/total

def test(model, text):
    with torch.no_grad():
        model.eval()
        encoded = encode_with_special(text).to(device)
        
        #Pad text to the length the model was trained on
        pad_len = inp_padded.shape[1] - len(encoded)
        encoded = torch.cat([encoded, torch.zeros(pad_len, dtype=torch.long)])
        
        #Gives it batch size 1
        encoded = encoded.unsqueeze(0)
        
        #Context vector
        hidden, cell = model.encoder(encoded)

        predicted = []
        decoder_input = torch.tensor([vocab["<sos>"]], device=device)
        for step in range(len(text.split())+2):
            prediction, hidden, cell = model.decoder(decoder_input, hidden, cell)
            best_guess = prediction.argmax(1).item()
            if best_guess == vocab["<eos>"]:
                break

            #Reverse dictionary look up (find the first key corresponding to a value since values are unique)
            translated = next(k for k, v in vocab.items() if v == best_guess)
            
            predicted.append(translated)
            decoder_input = torch.tensor([best_guess], device=device)
        return(predicted)

#Training hyperparameters
num_epochs = 100
learning_rate = 1e-3
vocab_size = len(vocab)
encoder_embedding_size = 64
decoder_embedding_size = 64
hidden_size = 64
num_layers = 1

#Model hyperparameters
encoder_net = Encoder(vocab_size, encoder_embedding_size, hidden_size, num_layers).to(device)
decoder_net = Decoder(vocab_size, decoder_embedding_size, hidden_size, vocab_size, num_layers).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#Ignore <pad>
criterion = nn.CrossEntropyLoss(ignore_index=0)
#Combat overfitting
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

#TRAINING LOOP
#------------------------------------------------------------
test_str = "The cat swordfights the ferret"
tf_ratio = 1
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} / {num_epochs}")
    model.train()

    epoch_loss = 0.0
    with torch.set_grad_enabled(True):
      for src_batch, trg_batch in train_loader:
          src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
          optimizer.zero_grad()

          #Drop TF to 0 when models pretty much stabilized (lets model make own predictions)
          if epoch < 50:
            output = model(src_batch, trg_batch, max(0.1, tf_ratio))
          else:
            output = model(src_batch, trg_batch, 0)

          #Skip <sos>, then flatten the batch size and seq length dimensions into 1
          output = output[:, 1:].reshape(-1, len(vocab))
          target = trg_batch[:, 1:].reshape(-1)

          loss = criterion(output, target)
          loss.backward()

          #Norm=magnitude, clipping gradient to ensure they don't explode
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1)
          optimizer.step()
          epoch_loss += loss.item()

    print(f"{test_str} => {test(model, test_str)}")
    print(f"Train Loss: {epoch_loss/len(train_loader)}")
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss}")
    tf_ratio -= 0.02
    scheduler.step(val_loss)
#----------------------------------------------------------------------

#Save model
torch.save(model.state_dict(), "reversal_model.pt")

#Final testing:
print(f"Test Loss: {evaluate(model, test_loader, criterion, device)}")
print(f"Test Acc: {accuracy(model, test_loader, device)}")