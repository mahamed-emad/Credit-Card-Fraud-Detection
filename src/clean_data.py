import pandas as pd
from pathlib import Path
from itertools import chain
from sklearn.model_selection import train_test_split

def select(folder="split", extn="*.csv"):
    path = "data/" + folder + "/"
    path = Path(path)
    ## to read files name
    # for fl in path.iterdir():
    #     print(fl.name)
    return path.glob(extn)

def merge(files):
    files = [pd.read_csv(fl) for fl in files]
    return pd.concat(files, ignore_index=True)

def shuffle(data, random=42):
    return data.sample(frac=1, random_state=random).reset_index(drop=True)

def split(data, random=42):
    train, test = train_test_split(data, train_size=0.8, random_state=random, stratify=data["Class"])
    val, test = train_test_split(test, test_size=0.5, random_state=random, stratify=test["Class"])
    dic = {0:"train", 1:"val", 2:"test"}
    for i, dt in enumerate([train, val, test]):
        save(dt, name=dic[i])

def read(data="data"):
    return pd.read_csv(f"data/{data}.csv")

def save(data, name="data", change=False):
    if change: data.to_csv(f"{name}.csv", index=False)
    else: data.to_csv(f"data/{name}.csv", index=False)

def pipl_clean():
    data = chain(select(), select(folder="split_sample"))
    data = merge(data)
    data = shuffle(data)
    split(data)

if __name__ == "__main__":
    pipl_clean()