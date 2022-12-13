import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    for prefix1, csvfile, input_keys in (
        (
            "DGF", "fixed_area_results.csv", [
                "grid_step",
                "n_nodes",
                "n_cores",
                "single_precision",
                "subdomain_ratio",
            ],
        ),
        (
            "DGV", "variable_area_results.csv", [
                "x_length",
                "y_length",
                "grid_step",
                "n_nodes",
                "n_cores",
                "single_precision",
                "subdomain_ratio",
            ],
        )
    ):
        df = pd.read_csv(csvfile)
        for prefix2, target in (
            ("PT1", "time1"),
            ("PT2", "time2"),
        ):
            keys = input_keys + [target]
            train_set, test_set = train_test_split(df[keys], test_size=0.2, random_state=42)
            train_set.to_csv(f"data/input/{prefix1}_{prefix2}_train.csv", index=False)
            test_set.to_csv(f"data/input/{prefix1}_{prefix2}_test.csv", index=False)

        df_list = []
        for n in range(1, 140):
            df_temp = df.copy()
            df_temp["n_hours"] = n
            df_temp["time_tot"] = df_temp["time1"] + df_temp["time2"] * (n - 1)
            df_list.append(df_temp)
        df = pd.concat(df_list)[input_keys + ["n_hours", "time_tot"]]
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
        train_set.to_csv(f"data/input/{prefix1}_PTTOT_train.csv", index=False)
        test_set.to_csv(f"data/input/{prefix1}_PTTOT_test.csv", index=False)
