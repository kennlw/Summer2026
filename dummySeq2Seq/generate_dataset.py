import pandas as pd

noun_ls = ["cat", "dog", "man", "mouse", "chicken", "ferret", "scorpion"]

verb_ls = ["meet", "like", "dislike", "challenge", "befriend", "celebrate", "swordfight", 
           "question", "serenade", "bite", "hug", "defend", "save", "paint", "illustrate"]

#Sentence templates

def template_1(noun1, noun2, verb):
    return(f"The {noun1} {verb}s the {noun2}.")

def template_2(noun1, noun2, verb):
    return(f"Please tell the {noun1} and the {noun2} to {verb} me.")

def template_3(noun1, noun2, verb):
    return(f"Do not {verb} the {noun1} or the {noun2}.")

def template_4(noun1, noun2, verb):
    return(f"I wanted to {verb} the {noun1}, but a {noun2} stopped me.")

def template_5(noun1, noun2, verb):
    return(f"I wonder why the {noun1} loves to {verb} the {noun2}.")

def template_6(noun1, noun2, verb):
    return(f"The {noun1} does not {verb} the {noun2}")

#Generated sentences

res_ls = []

for noun_1 in noun_ls:
    for noun_2 in noun_ls[::-1]:
        for verb in verb_ls:
            res_ls.append(template_1(noun_1, noun_2, verb))
            res_ls.append(template_2(noun_1, noun_2, verb))
            res_ls.append(template_3(noun_1, noun_2, verb))
            res_ls.append(template_4(noun_1, noun_2, verb))
            res_ls.append(template_5(noun_1, noun_2, verb))

#Cleaned sentences

src_ls = []
trg_ls = []

for sent in res_ls:
    new_sent = []
    for word in sent.split():
        if any(c in word for c in [".", ","]):
            new_sent.append(word[:-1])
        else:
            new_sent.append(word)
    src_ls.append(" ".join(new_sent))
    new_sent.reverse()
    trg_ls.append(" ".join(new_sent))

#Create and save as CSV

df = pd.DataFrame({"source": src_ls, "target": trg_ls})
df.to_csv('new_dataset.csv', index=False)