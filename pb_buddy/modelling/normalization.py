import csv
import gzip
import io
from abc import ABC, abstractmethod

import pandas as pd
import requests


class CPISource(ABC):
    @abstractmethod
    def get_cpi_data(self) -> pd.DataFrame:
        pass

    def _process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["year"] = df["year"].astype(int)
        df["cpi"] = df["cpi"].astype(float)
        df["most_recent_cpi"] = df["most_recent_cpi"].astype(float)
        df["currency"] = df["currency"].astype(str)
        return df


class USCPISource(CPISource):
    def get_cpi_data(self) -> pd.DataFrame:
        headers = {
            "Host": "download.bls.gov",
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
            "Accept-Language": "en-US",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://download.bls.gov/pub/time.series/cu/",
        }
        df_us_cpi_data = (
            pd.read_csv(
                "https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems",
                sep="\t",
                header=None,
                skiprows=1,
                names=["series_id", "year", "period", "cpi", "footnote"],
                storage_options=headers,
            )
            .assign(series_id=lambda _df: _df.series_id.str.strip())
            .filter(["year", "cpi", "series_id"])
            .query("series_id == 'CUSR0000SA0'")
            .groupby("year", as_index=False)
            .mean()
            .assign(most_recent_cpi=lambda x: x.loc[x.year.idxmax(), "cpi"], currency="USD")
        )
        df_us_cpi_data = self._process_data(df_us_cpi_data)
        return df_us_cpi_data


class CanadaCPISource(CPISource):
    def get_cpi_data(self) -> pd.DataFrame:
        start_year = "2000"
        end_year = str(pd.Timestamp.now().year)
        df_can_cpi_data = pd.read_csv(
            f"https://www150.statcan.gc.ca/t1/tbl1/en/dtl!downloadDbLoadingData-nonTraduit.action?pid=1810000501&latestN=0&startDate={start_year}0101&endDate={end_year}0101&csvLocale=en&selectedMembers=%5B%5B2%5D%2C%5B2%2C3%2C79%2C96%2C139%2C176%2C184%2C201%2C219%2C256%2C274%2C282%2C285%2C287%2C288%5D%5D&checkedLevels="
        )
        df_can_cpi_data = (
            df_can_cpi_data.query("`Products and product groups`=='All-items'")
            .filter(["REF_DATE", "VALUE"])
            .rename(columns={"REF_DATE": "year", "VALUE": "cpi"})
            .assign(most_recent_cpi=lambda x: x.loc[x.year.idxmax(), "cpi"], currency="CAD")
        )
        df_can_cpi_data = self._process_data(df_can_cpi_data)
        return df_can_cpi_data


class EuroCPISource(CPISource):
    def get_cpi_data(self) -> pd.DataFrame:
        url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/tec00027?format=TSV&compressed=true"
        response = requests.get(url)
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            df_eurostat = pd.read_csv(f, delimiter="\t")
        df_euro_cpi_data = (
            df_eurostat.rename(columns={"freq,unit,coicop,geo\TIME_PERIOD": "region"})
            .query("region == 'A,INX_A_AVG,CP00,EA19'")
            .melt(id_vars=["region"], var_name="year", value_name="cpi")
            .assign(currency="EUR", most_recent_cpi=lambda _df: _df["cpi"].dropna().iloc[-1])
            .drop(columns=["region"])
        )
        df_euro_cpi_data = self._process_data(df_euro_cpi_data)
        return df_euro_cpi_data


class UKCPISource(CPISource):
    def get_cpi_data(self) -> pd.DataFrame:
        response = requests.get(
            "https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/d7bt/mm23"
        )
        data = list(csv.reader(io.StringIO(response.text)))
        df = pd.DataFrame(data[1:], columns=data[0])
        important_notes = df[df["Title"] == "Important notes"]
        quarterly_data = df[df["Title"].str.contains("Q")]
        df_uk_cpi_data = (
            df.loc[important_notes.index[0] + 1 : quarterly_data.index[0] - 1]
            .rename(columns={"Title": "year", "CPI INDEX 00: ALL ITEMS 2015=100": "cpi"})
            .assign(currency="GBP", most_recent_cpi=lambda _df: _df["cpi"].dropna().iloc[-1])
        )
        df_uk_cpi_data = self._process_data(df_uk_cpi_data)
        return df_uk_cpi_data


class CPISourceFactory:
    sources = {
        "united-states": USCPISource,
        "canada": CanadaCPISource,
        "eu": EuroCPISource,
        "uk": UKCPISource,
    }

    def get_source(self, region: str) -> CPISource:
        try:
            return self.sources[region]()
        except KeyError:
            raise ValueError(f"Region must be one of {list(self.sources.keys())}")
