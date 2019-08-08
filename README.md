# futsal

Forecasting the future level of sales for a certain chain of stores.

# Installation

To install and use this package in its full potential you need to clone the repository and place the `store.csv` and `train.csv` files into `_data` directory (if not existing, you can ).

```bash
cd /path/to/your/git/repos/destination/
git clone https://github.com/mrtovsky/futsal.git

cd futsal && mkdir _data
mv /path/to/your/csvs/*.csv /path/to/your/git/repos/destination/futsal/_data/
```

After that you should install the futsal package, either directly from the previously cloned repository or from github.
You can always consider runing this installation with `--no-deps` flag, to not override your current packages, if upgrade is needed.

```bash
pip install git+https://github.com/mrtovsky/futsal.git
```

When `.csv` files are on its places, we can run our data preparation with simultaneous path correctness validation.

```bash
python -m futsal /path/to/your/git/repos/destination/futsal/ prepare_store --validate
```

If you want to know what arguments could be passed to the **futsal** main, then write:

```bash
python -m futsal --help
```
