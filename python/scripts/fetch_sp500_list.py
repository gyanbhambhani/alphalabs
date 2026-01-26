"""
Fetch S&P 500 Ticker List

Retrieves the current list of S&P 500 constituents from Wikipedia
and saves it with metadata (sector, industry, market cap, etc.)
"""
import pandas as pd
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from io import StringIO


@dataclass
class StockInfo:
    """Stock information"""
    symbol: str
    name: str
    sector: str
    sub_industry: str
    headquarters: str
    date_added: str
    cik: str
    founded: str


def fetch_sp500_list() -> pd.DataFrame:
    """
    Fetch S&P 500 list from Wikipedia.
    
    Returns:
        DataFrame with columns: Symbol, Security, GICS Sector, 
        GICS Sub-Industry, Headquarters Location, Date added, 
        CIK, Founded
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    try:
        # Add headers to avoid 403 Forbidden error
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) '
                         'Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Read the first table from Wikipedia with headers
        response = requests.get(url, headers=headers)
        tables = pd.read_html(StringIO(response.text))
        sp500_table = tables[0]
        
        print(f"✓ Fetched {len(sp500_table)} stocks from Wikipedia")
        return sp500_table
    
    except Exception as e:
        print(f"✗ Error fetching S&P 500 list: {e}")
        raise


def clean_sp500_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the S&P 500 data.
    
    Args:
        df: Raw dataframe from Wikipedia
        
    Returns:
        Cleaned dataframe
    """
    # Standardize column names
    df.columns = df.columns.str.strip()
    
    # Rename for consistency
    column_mapping = {
        'Symbol': 'symbol',
        'Security': 'name',
        'GICS Sector': 'sector',
        'GICS Sub-Industry': 'sub_industry',
        'Headquarters Location': 'headquarters',
        'Date added': 'date_added',
        'CIK': 'cik',
        'Founded': 'founded'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Clean symbols (remove dots, handle special characters)
    df['symbol'] = df['symbol'].str.replace('.', '-')
    
    # Fill NaN values
    df = df.fillna('')
    
    # Sort by symbol
    df = df.sort_values('symbol')
    
    return df


def get_unique_sectors(df: pd.DataFrame) -> List[str]:
    """Get list of unique sectors"""
    return sorted(df['sector'].unique().tolist())


def get_stocks_by_sector(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Filter stocks by sector"""
    return df[df['sector'] == sector]


def save_sp500_list(df: pd.DataFrame, filepath: str = "./sp500_list.csv"):
    """Save S&P 500 list to CSV"""
    df.to_csv(filepath, index=False)
    print(f"✓ Saved S&P 500 list to {filepath}")


def load_sp500_list(filepath: str = "./sp500_list.csv") -> Optional[pd.DataFrame]:
    """Load S&P 500 list from CSV"""
    path = Path(filepath)
    if path.exists():
        return pd.read_csv(filepath)
    return None


def main():
    """Fetch and save S&P 500 list"""
    print("=" * 70)
    print("S&P 500 Ticker List Fetcher")
    print("=" * 70)
    print()
    
    print("[1/3] Fetching S&P 500 list from Wikipedia...")
    df = fetch_sp500_list()
    print()
    
    print("[2/3] Cleaning data...")
    df = clean_sp500_data(df)
    print(f"  ✓ {len(df)} stocks processed")
    print()
    
    print("[3/3] Analyzing data...")
    sectors = get_unique_sectors(df)
    print(f"  Sectors found: {len(sectors)}")
    for sector in sectors:
        count = len(get_stocks_by_sector(df, sector))
        print(f"    • {sector}: {count} stocks")
    print()
    
    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "sp500_list.csv"
    save_sp500_list(df, str(output_path))
    
    print("=" * 70)
    print("✓ S&P 500 list successfully fetched and saved!")
    print(f"  Total stocks: {len(df)}")
    print(f"  Output: {output_path}")
    print("=" * 70)
    print()
    
    # Show sample
    print("Sample (first 10 stocks):")
    print(df[['symbol', 'name', 'sector']].head(10).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
