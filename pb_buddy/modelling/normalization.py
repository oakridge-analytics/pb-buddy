import csv
import gzip
import io
from abc import ABC, abstractmethod

import pandas as pd
import requests


def get_cpi_data(region=None):
    """
    Get the Consumer Price Index (CPI) data for the specified region.
    All regions returned if not specified.

    Parameters
    ----------
    region : str, optional
        The region for which to get the CPI data. If None, the data for the united-states is returned.
        Options: "united-states", "canada", "uk", "eu"

    Returns
    -------
    pd.DataFrame
        The CPI data for the specified region.
    """
    accepted_regions = ["united-states", "canada", "uk", "eu"]
    if region is not None and region not in accepted_regions:
        raise ValueError(f"Region must be one of {accepted_regions}")

    # Load the CPI data
    cpi_data = pd.read_csv("data/cpi_data.csv")

    # Filter the data for the specified region
    if region is not None:
        cpi_data = cpi_data[cpi_data["Region"] == region]

    return cpi_data


class CPISource(ABC):
    @abstractmethod
    def get_cpi_data(self) -> pd.DataFrame:
        pass


class USCPISource(CPISource):
    def get_cpi_data(self) -> pd.DataFrame:
        """Returns dataframe with columns: year, cpi, most_recent_cpi, currency"""
        headers = {
            "Host": "download.bls.gov",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/113.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-CA,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://download.bls.gov/pub/time.series/cu/",
        }
        # Based on links found here: https://github.com/palewire/cpi/blob/master/cpi/download.py
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
        return df_us_cpi_data


class CanadaCPISource(CPISource):
    def get_cpi_data(self) -> pd.DataFrame:
        """Returns dataframe with columns: year, cpi, most_recent_cpi, currency"""
        # Canada data
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
        return df_can_cpi_data


class EuroCPISource(CPISource):
    def get_cpi_data(self) -> pd.DataFrame:
        """Returns dataframe with columns: year, cpi, most_recent_cpi, currency"""
        url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/tec00027?format=TSV&compressed=true"
        response = requests.get(url)
        with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
            df_eurostat = pd.read_csv(f, delimiter="\t")
        return (
            # Filter to Euro area, from 2015-2022 that had 19 countries
            df_eurostat.rename(columns={"freq,unit,coicop,geo\TIME_PERIOD": "region"})
            .query("region == 'A,INX_A_AVG,CP00,EA19'")
            .melt(id_vars=["region"], var_name="year", value_name="cpi")
            .assign(currency="EUR", most_recent_cpi=lambda _df: _df.loc[_df["cpi"].last_valid_index()]["cpi"])
            .drop(columns=["region"])
        )


class UKCPISource(CPISource):
    def get_cpi_data(self) -> pd.DataFrame:
        response = requests.get(
            "https://www.ons.gov.uk/generator?format=csv&uri=/economy/inflationandpriceindices/timeseries/d7bt/mm23"
        )
        data = list(csv.reader(io.StringIO(response.text)))
        df = pd.DataFrame(data[1:], columns=data[0])
        # Find first row where "Important notes" is mentioned
        important_notes = df[df["Title"] == "Important notes"]
        # Find first index where "Title" column has a "Q" in it
        quarterly_data = df[df["Title"].str.contains("Q")]
        return (
            df.loc[important_notes.index[0] + 1 : quarterly_data.index[0] - 1]
            .rename(columns={"Title": "year", "CPI INDEX 00: ALL ITEMS 2015=100": "cpi"})
            .assign(currency="GBP", most_recent_cpi=lambda _df: _df.loc[_df["cpi"].last_valid_index()]["cpi"])
        )


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
