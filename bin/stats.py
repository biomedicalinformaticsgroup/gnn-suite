#!/usr/bin/env python3
import typer
from pathlib import Path
from typing import List 
import pandas as pd

app = typer.Typer()

@app.command()
def compute(filename: str, data: List[Path], model: str):
    df = [pd.read_table(f, delimiter="\s+", comment='#') for f in data]
    data = pd.concat(df)
    data.insert(0, 'model', model)
    data.groupby(['model', 'epoch']).mean().tail(1).to_csv(filename, sep=' ')


@app.command()
def collect(filename: str, data: List[Path]):
    df = [pd.read_table(f, delimiter="\s+", comment='#') for f in data]
    data = pd.concat(df)
    data.to_latex(filename, index=False, columns=['model', 'prec', 'rec', 'acc', 'bacc', 'auc'])


if __name__ == "__main__":
    app()
