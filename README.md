## Machine Learning Final Project (Traffic Data)

This repository contains the setup and starter notebook(s) for the Machine Learning final project analyzing NYC traffic / mobility (NYC TLC trip-level) data, with a focus on congestion pricing impacts.

## Data download (Google Drive)

The main dataset file is expected at the project root as:

- `nyc_taxi_ml_dataset_2024_2025.parquet` (ignored by git via `.gitignore`)

- Open the shared Google Drive link for the dataset.
- Download the parquet file.
- Place it in this folder (project root) and ensure it is named exactly `nyc_taxi_ml_dataset_2024_2025.parquet`.


## Python environment setup

### Create and activate a virtual environment (macOS / zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run the notebook

- Open `load_parquet_dataset.ipynb` in Jupyter and run the cells to verify the dataset loads.
- If Jupyter is not installed in your environment:

```bash
python -m pip install jupyter
jupyter lab
```

