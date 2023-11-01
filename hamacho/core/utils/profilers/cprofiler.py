"""Profiling using cProfiler"""

import os
import re
import pstats

import click
import numpy as np
import pandas as pd

from io import StringIO
from pathlib import Path
from cProfile import Profile
from typing import Literal, Union


class HamachoProfiler:
    """
    Profiler that uses the built-in cProfile and saves
    output in excel format
    """

    def __init__(
        self,
        save_path: Union[str, Path],
        type: Literal["custom", "simple", "advanced", "expert"] = "simple",
        sort_by: Literal["cumtime", "tottime", "percall", "function"] = "cumtime",
        sort_ascending: bool = False,
        **kwargs,
    ) -> None:
        self.save_path = save_path
        self.type = type
        if sort_by == "function":
            sort_by = "filename:lineno(function)"
        else:
            sort_by = sort_by + "(s)" 
        self.sort_by = sort_by
        self.sort_ascending = sort_ascending
        self.filter = ""
        self.pr_output = StringIO()
        if self.type == "custom":
            self.custom_filters()
        else:
            self.set_filters()
            self.profile_built_ins = self.profile_built_ins \
                                    if "builtins" not in kwargs else kwargs["builtins"]
        self.pr = Profile(**kwargs)

    def set_filters(self) -> None:
        self.filter = ""
        hmc = Path('hamacho')
        if self.type == "simple":
            self.filter = f"{hmc / 'core'}|{hmc / 'plug_in'}|dataloader"
            self.profile_built_ins = False
        elif self.type == "advanced":
            self.filter = ""
            self.profile_built_ins = False
        elif self.type == "expert":
            self.filter = ""
            self.profile_built_ins = True

    def custom_filters(self) -> None:
        func_path = os.path.join(os.path.dirname(__file__), "custom_profiling", "functions.txt")
        module_path = os.path.join(os.path.dirname(__file__), "custom_profiling", "modules.txt")

        if not os.path.exists(func_path) or not os.path.exists(module_path):
            ValueError(
                f"functions.txt or modules.txt files must be available in"
                "./hamacho/core/utils/profilers/custom_profiling/ if profiling type is 'custom'"
            )
        funcs = ()
        if os.path.exists(func_path):
            with open(func_path, "r") as f:
                funcs_txt = f.read()
            funcs = tuple(line for line in funcs_txt.splitlines()
                          if line and not line.startswith("#"))
        modules = ()
        if os.path.exists(module_path):
            with open(module_path, "r") as f:
                modules_txt = f.read()
            modules = tuple(line for line in modules_txt.splitlines()
                            if line and not line.startswith("#"))

        functions_modules = (*funcs, *modules)
        self.filter = "|".join(functions_modules)
        self.profile_built_ins = False

    def paste_output(self) -> None:
        stats = pstats.Stats(self.pr, stream=self.pr_output)
        # stats.sort_stats(self.sort_by)
        stats.print_stats(self.filter)

    def to_csv_format(self, output: str) -> str:
        formated = ""
        for line in output.splitlines():
            line = re.sub("[ ]+", ";", " " + line, count=6)
            formated += line + "\n"

        return formated

    def to_pandas(self, csv_as_str: str) -> pd.DataFrame:
        df = pd.read_csv(
            StringIO(csv_as_str),
            skiprows=6,
            sep=";",
            names=(
                "ncalls", "tottime(s)", "_percall", "cumtime(s)",
                "percall(s)", "filename:lineno(function)",
            ),
        )
        df.set_index(np.arange(len(df)), inplace=True)
        df.drop("_percall", inplace=True, axis=1)
        
        df["filepath"] = self._get_filepaths(df, "filename:lineno(function)")
        df["filename:lineno(function)"] = self._get_filenames(df, "filename:lineno(function)")
        df.sort_values(by=self.sort_by, ascending=self.sort_ascending, inplace=True)
        if self.type == "expert":
            df.loc["Total"] = pd.Series(df["tottime"].sum(), index=["tottime"])

        return df

    def parse_stats(
        self,
    ) -> pd.DataFrame:
        self.paste_output()
        output = self.pr_output.getvalue()
        # with open("test.txt", "w") as f:
        #     f.write(output)
        csv_str = self.to_csv_format(output)
        df = self.to_pandas(csv_str)
        return df

    def save_stats(
        self,
        df: pd.DataFrame
    ) -> None:
        filepath = Path(self.save_path) / f"{self.type}_profiler.xlsx"
        writer = pd.ExcelWriter(filepath, engine="xlsxwriter")
        sheet_name = f"{self.type}-report"
        df.to_excel(writer, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        filename_col = "filename:lineno(function)"
        filepath_col = "filepath"
        filename_col_width = max(
            df[filename_col].astype(str).map(len).max(), 
            len(filename_col))
        filepath_col_width = max(
            df[filepath_col].astype(str).map(len).max(), 
            len(filepath_col))
        filename_col_idx = df.columns.get_loc(filename_col) + 1
        filepath_col_idx = df.columns.get_loc(filepath_col) + 1
        worksheet.set_column(filename_col_idx, filename_col_idx, filename_col_width)
        worksheet.set_column(filepath_col_idx, filepath_col_idx, filepath_col_width)
        writer.save()
        click.secho(f"INFO: profiling results saved at {filepath}")

    def enable(
        self,
    ) -> None:
        self.pr.enable()

    def disable(
        self,
    ) -> None:
        self.pr.disable()

    def _get_filepaths(
        self,
        df: pd.DataFrame,
        col_name: str,
    ) -> pd.Series:
        s = df.apply(
            lambda row: os.path.dirname(
                row[col_name]
            ) if isinstance(row[col_name], str) and ".py" in row[col_name] else "Not Applicable",
            axis=1,
        )
        return s

    def _get_filenames(
        self,
        df: pd.DataFrame,
        col_name: str,
    ) -> pd.Series:
        s = df.apply(
            lambda row: os.path.basename(
                row[col_name]
            ) if isinstance(row[col_name], str) and ".py" in row[col_name] else row[col_name],
            axis=1,
        )
        return s

    def __enter__(
        self
    ) -> Profile:
        self.pr.enable()
        return self.pr

    def __exit__(
        self
    ) -> None:
        self.pr.disable()
        df = self.parse_stats()
        self.save_stats(df)
