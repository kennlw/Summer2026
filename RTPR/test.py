import torch
from torch import nn
import pandas as pd
import random

sos_val = 2
eos_val = 3

vocab_size = 61

left_role = 0
right_role = 1

# tpr_df = pd.read_csv("reversal_data.csv")
tpr_df = pd.read_csv("identity_data.csv")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tpr_df["source"] = tpr_df["source"].apply(lambda x: eval(x))
# tpr_df['target'] = tpr_df['target'].apply(lambda x: eval(x))
tpr_df["context_vector"] = tpr_df["context_vector"].apply(lambda x: eval(x))

df_sorted = tpr_df.sort_values(by="source", key=lambda col: col.map(len))
model_size = len(df_sorted["context_vector"].iloc[0])


num_df = pd.read_csv("num_dataset.csv")
final_targets = [torch.tensor([sos_val] + eval(s) + [eos_val], dtype=torch.long) for s in num_df["source"]]


def right_recurse(lst):
    batch_size = lst[0].shape[0]
    tensors = [[x, torch.full((batch_size,), left_role)] for x in lst]
    
    acc = [tensors[-1]]
    for t in reversed(tensors[:-1]):
        acc = [t, [acc, torch.full((batch_size,), right_role)]]
    return acc

def left_recurse( lst):
    batch_size = lst[0].shape[0]
    tensors = [[x, torch.full((batch_size,), right_role)] for x in lst]
    
    acc = [tensors[0]]
    for t in tensors[1:]:
        acc = [[acc, torch.full((batch_size,), left_role)], t]
    return acc

def collate_tensors(tensor):
    return [tensor[:, i] for i in range(tensor.shape[1])]


batch_size = 5

sources = df_sorted["source"].tolist()
targets = df_sorted["context_vector"].tolist()

#rev
# reversed_list = df_sorted["target"].tolist()

src_batches = []
trg_batches = []

#rev
# rev_batches = []

for i in range(0, len(df_sorted), batch_size):
    # print(i)
    src_batch = sources[i:i + batch_size]   # list of 50 lists (each list is a source sequence)
    trg_batch = targets[i:i + batch_size]   # list of 50 lists (each list is a context vector)
    # rev_batch = reversed_list[i:i + batch_size]   # list of 50 lists (each list is a source sequence)

    # collate_lists expects *lists unpacked, where each list is one sequence's tokens
    # so we unpack the 50 source sequences — gives one tensor per position across the batch

    src_batches.append(src_batch)
    # rev_batches.append(rev_batch)
    trg_batches.append(torch.stack([torch.tensor(v, dtype=torch.float32) for v in trg_batch]))  # (50, model_size)

# 80/10/10 split at batch level
n       = len(src_batches)        # 120
n_train = int(0.8 * n)            # 96
n_val   = int(0.1 * n)            # 12
# test: remainder                 # 12

combined = list(zip(src_batches, trg_batches))
# combined = list(zip(src_batches, trg_batches, rev_batches))
random.shuffle(combined)
src_batches, trg_batches = zip(*combined)

src_batches = [torch.tensor(s, dtype=torch.long) for s in src_batches]
# rev_batches = [torch.tensor(s, dtype=torch.long) for s in rev_batches]

class RTPREncode(nn.Module):
    def __init__(self, nfillers, nroles, filler_size, role_size, batch_size, model_size):
        super(RTPREncode, self).__init__()
        self.fillers = nn.Embedding(nfillers, filler_size)
        self.roles = nn.Embedding(nroles, role_size)
        self.filler_size = filler_size
        self.role_size = role_size
        self.model_size = model_size
        self.batch_size = batch_size
        self.dropout = nn.Dropout(0.4)
        self.w = torch.nn.Parameter(torch.randn(filler_size, filler_size*role_size)*0.01)
        self.wo = torch.nn.Parameter(torch.randn(model_size, filler_size)*0.01)

    def forward(self, x, recurse = False):
        result = torch.zeros(self.batch_size, self.filler_size, self.role_size, device=self.w.device)
        if recurse:
            collate = collate_tensors(x)
            x = right_recurse(collate)
        for pair in x:
            filler = pair[0]
            role = pair[1]
            
            if isinstance(filler, list):
                filler = self(filler)
            else:
                filler = self.fillers(filler)  # (batch_size, filler_size)
            
            role = self.roles(role)  # (batch_size, role_size)
            result += torch.einsum("bf,br->bfr", filler, role)
        
        result = result.view(self.batch_size, -1)  
        result = torch.einsum("ij,bj->bi", self.w, result)  
        return result

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
        self.dropout_out = nn.Dropout(0.3)

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

    def forward(self, context, source):
        batch_size = source.shape[0]
        target_len = source.shape[1]

        #Tensor of zeros with shape (batch_size, target_len, vocab_size), will gradually be filled in by the decoder
        #BN: Vocab_size != Embedding_Size
        outputs = torch.zeros(batch_size, target_len, vocab_size).to(device)

        #Context vector
        hidden, cell = context.split(128, dim=1)
        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)

        #Effectively the <sos> token
        x = torch.full((batch_size,), 2)

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)
            #Fill in the outputs variable
            outputs[:, t] = output

            #Chooses the one with the highest probability
            best_guess = output.argmax(1)

            #Using random to ensure teacher forcing is applied only a certain percentage of the time
            x = best_guess

        return outputs
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tpr_model = RTPREncode(61, 2, 128, 64, 5, 256)

encoder_embedding_size = 64
decoder_embedding_size = 64
hidden_size = 128
num_layers = 1

#Model hyperparameters
encoder_net = Encoder(vocab_size, encoder_embedding_size, hidden_size, num_layers).to(device)
decoder_net = Decoder(vocab_size, decoder_embedding_size, hidden_size, vocab_size, num_layers).to(device)

s2s_model = Seq2Seq(encoder_net, decoder_net).to(device)
s2s_model.load_state_dict(torch.load("num_identity_model.pt", weights_only=True, map_location=torch.device('cpu')))
tpr_model.load_state_dict(torch.load("RBT_id.pt", weights_only=True, map_location=torch.device('cpu')))

with torch.no_grad():
    tpr_model.eval()
    s2s_model.eval()
    total = 0
    for i in range(len(src_batches)):
        src_batch = src_batches[i]
        if ((src_batch.shape[1]) - 2) == 9:
            continue
        encoding = tpr_model(src_batch, recurse=True)
        encoder_output = torch.matmul(encoding, tpr_model.wo.T)
        hidden, cell = encoder_output.split(128, dim=1)
        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)
        
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        predicted = [[] for _ in range(batch_size)]
        decoder_input = torch.full((batch_size,), sos_val, dtype=torch.long, device=device)

        for step in range(src_batch.shape[1]-2):
            prediction, hidden, cell = s2s_model.decoder(decoder_input, hidden, cell)
            best_guesses = prediction.argmax(1) 

            for k in range(batch_size):
                if active[k]:
                    guess = best_guesses[k].item()
                    if guess == eos_val:
                        active[k] = False
                    else:
                        predicted[k].append(guess)

            if not active.any():
                break

            decoder_input = best_guesses
        
        #In case eos too early
        max_len = (src_batch.shape[1]) - 2
        predicted = [lst + [0] * (max_len - len(lst)) for lst in predicted]
        output = torch.tensor(predicted, dtype=torch.long)

        #identity
        src_batch = src_batch[:, 1:-1]
        print(output)
        # print(src_batch)
        total += (src_batch == output).float().mean()

        #reversal
        # trg_batch = rev_batches[i][:, 1:-1]
        # total += (trg_batch == output).float().mean()
    print(f"Average accuracy: {total/len(src_batches)}")
