import argparse
import os
import pathlib

from datetime import timedelta

import numpy as np
import pandas as pd


def prepare_store(args):
    """Prepare and pickle the store.csv file."""
    data_path = args.path.joinpath("_data")

    store = pd.read_csv(
        data_path.joinpath("store.csv").resolve(), low_memory=False
    )

    # Impute missing values with the corresponding store type mean.
    store_comp_distance = store.groupby("StoreType")["CompetitionDistance"]
    store["CompetitionDistance"] = store_comp_distance.transform(
        lambda x: x.fillna(np.round(x.mean(), decimals=-1))
    )

    # Concatenate into one variable the competition opening and promo2 date.
    store["CompetitionOpenSinceDate"] = pd.to_datetime(
        store.CompetitionOpenSinceYear * 100 + store.CompetitionOpenSinceMonth,
        format="%Y%m"
    )
    store["Promo2SinceDate"] = pd.to_datetime(
        store["Promo2SinceYear"] * 1000 + store["Promo2SinceWeek"] * 10 + 1,
        format="%Y%W%w"
    )

    # Fill NaNs with a date {from a long time ago, from future} respectively.
    store["CompetitionOpenSinceDate"].fillna(
        pd.Timestamp(year=1900, month=1, day=1),
        inplace=True
    )
    store["Promo2SinceDate"].fillna(
        pd.Timestamp(year=2100, month=1, day=1),
        inplace=True
    )

    # Encode categorical `PromoInterval` feature (-1 equals NaN).
    store["PromoIntervalType"] = store["PromoInterval"].apply(
        _encode_promo_interval
    )

    _export_to_pickle(store, data_path, "store.pkl")


def prepare_train(args):
    """Prepare and pickle the train.csv file."""
    data_path = args.path.joinpath("_data")
    
    train = pd.read_csv(
        data_path.joinpath("train.csv").resolve(), low_memory=False
    )

    train["Date"] = pd.to_datetime(train["Date"])
    
    # Add lagged variables.
    train_lagged = train[[
        "Store", "Date", "Customers", "Open", 
        "Promo", "StateHoliday", "SchoolHoliday"
    ]].copy()
    train_lagged["Date"] = train_lagged["Date"] + timedelta(days=1)
    train_lagged.rename(
        columns={
            "Customers": "CustomersYesterday",
            "StateHoliday": "StateHolidayYesterday",
            "SchoolHoliday": "SchoolHolidayYesterday",
            "Open": "OpenYesterday",
            "Promo": "PromoYesterday"
        },
        inplace=True
    )

    # Add accelerated variables.
    train_accel = train[[
        "Store", "Date", "Open", 
        "StateHoliday", "SchoolHoliday"
    ]].copy()
    train_accel["Date"] = train_accel["Date"] - timedelta(days=1)
    train_accel.rename(
        columns={
            "Open": "OpenTomorrow",
            "StateHoliday": "StateHolidayTomorrow",
            "SchoolHoliday": "SchoolHolidayTomorrow"
        },
        inplace=True
    )

    train = train.merge(
        train_lagged, how="inner", on=["Store", "Date"]
    )
    train = train.merge(
        train_accel, how="inner", on=["Store", "Date"]
    )

    train.drop(
        ["Customers", "Year", "Month", "Day"],
        axis="columns", 
        inplace=True
    )

    # Add variable liquidating seasonal effect.
    train["ChristmasInAWeek"] = train["Date"].apply(_is_christmas_soon)

    _export_to_pickle(train, data_path, "train.pkl")


def create_panel(args):
    """Merges `store` and `train` files and makes feature engineering."""
    data_path = args.path.joinpath("_data")

    is_store = os.path.exists(data_path.joinpath("store.pkl"))
    is_train = os.path.exists(data_path.joinpath("train.pkl"))

    assert is_store and is_train, (
        "Execute 'prepare_store' and 'prepare_train' before calling this "
        "function!"
    )

    store = pd.read_pickle(data_path.joinpath("store.pkl"))
    train = pd.read_pickle(data_path.joinpath("train.pkl"))

    panel = train.merge(
        store.drop(
            [
                "Sales",
                "CompetitionOpenSinceMonth",
                "CompetitionOpenSinceYear",
                "Promo2",
                "Promo2SinceWeek",
                "Promo2SinceYear",
                "PromoInterval"
            ],
            axis="columns"
        ),
        how="left", 
        on="Store"
    )

    # Convert 3 to 0 to match the modulo function.
    panel.loc[panel["PromoIntervalType"] == 3, "PromoIntervalType"] = 0

    # Assign 1 where promo2 is already active.
    panel["Promo2Active"] = 0
    panel.loc[
        ((panel["Date"] - panel["Promo2SinceDate"]).astype(int) > 0) 
        & (panel["Date"].dt.month % 3 == panel["PromoIntervalType"]), 
        "Promo2Active"
    ] = 1

    panel["CompetitionActive"] = np.where(
        (panel["Date"] - panel["CompetitionOpenSinceDate"]).astype(int) > 0, 
        1,
        0
    )
    panel.drop(
        [
            "CompetitionOpenSinceDate",
            "Promo2SinceDate",
            "PromoIntervalType"
        ],
        axis="columns",
        inplace=True
    )
    panel = pd.get_dummies(
        data=panel,
        columns=[
            "DayOfWeek",
            "StateHoliday",
            "StateHolidayYesterday",
            "StateHolidayTomorrow",
            "StoreType",
            "Assortment"
        ],
        drop_first=True
    )
    panel = panel[panel["Open"] == 1]

    _export_to_pickle(panel, data_path, "panel.pkl")


def fit(args):
    """Fit the panel data regression."""
    print(
        "For now this step is only implemented in the: {}"
        .format(args.path.joinpath("notebooks", "modeling.ipynb"))
    )


def _export_to_pickle(data, path, name):
    """Pickles data and exports to the given path."""
    if not os.path.exists(path.joinpath(name).resolve()):
        data.to_pickle(path.joinpath(name).resolve())
    else:
        if not os.path.exists(path.joinpath("_temp").resolve()):
            os.makedirs(path.joinpath("_temp").resolve())
        data.to_pickle(path.joinpath("_temp", name).resolve())
        print(
            "File '{0}' already exists in the location: {1}\n"
            "I am saving the '{0}' file to the: {2}\n"
            "Before executing the 'fit' action make sure to place the "
            "just prepared '{0}' file to the main 'data' location."
            .format(
                name,
                str(path.resolve()),
                str(path.joinpath("_temp").resolve())
            )
        )


def _encode_promo_interval(cell):
    if pd.isna(cell):
        return -1
    elif cell.startswith("Jan"):
        return 1
    elif cell.startswith("Feb"):
        return 2
    else:
        return 3


def _is_christmas_soon(col):
    current_year = col.year
    nicholas = pd.Timestamp(
        year=current_year,
        month=12,
        day=6
    )
    christmas = pd.Timestamp(
        year=current_year,
        month=12,
        day=24
    )
    nicholas_counter = len(pd.date_range(
        start=col, end=nicholas, freq="D"
    ))
    christmas_counter = len(pd.date_range(
        start=col, end=christmas, freq="D"
    ))
    is_nicholas_soon = 0 < nicholas_counter <= 7
    is_christmas_soon = 0 < christmas_counter <= 7
    if is_nicholas_soon or is_christmas_soon:
        return 1
    else:
        return 0


def _add_bool_arg(parser, name, default=True, help_comment=""):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--" + name, dest=name, action="store_true", help=help_comment
    )
    group.add_argument(
        "--no-" + name, dest=name, action="store_false", help=help_comment
    )
    parser.set_defaults(**{name:default})


def main():
    parser = argparse.ArgumentParser(
        description="Control the process of forecasting future sales"
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the `futsal` repository root"
    )
    parser.add_argument(
        "action",
        type=str,
        choices=["prepare_store", "prepare_train", "create_panel", "fit"],
        help="Action name to execute"
    )
    _add_bool_arg(parser, "validate", help_comment="Validate specified path")

    args = parser.parse_args()

    args.path = pathlib.Path(args.path)

    if args.validate:
        is_existing = os.path.exists(args.path)
        is_repo_root = os.path.split(args.path)[-1] == "futsal"
        is_git_repo = os.path.exists(args.path.joinpath(".git"))

        assert is_existing and is_repo_root and is_git_repo, (
            "Path specified should be adressing the highest folder created "
            "after cloning the repository."
        )

    # Call the specified function.
    globals()[args.action](args)


if __name__ == "__main__":
    main()
