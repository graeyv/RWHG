import pandas as pd
from pathlib import Path

# Base path is the root of your project, i.e., thesis_project/
BASE_DIR = Path(__file__).resolve().parents[1]
CLEAN_DIR = BASE_DIR / "data" / "clean"
MAP_DIR = BASE_DIR / "data" / "mappings"

## --- filter affair & councillor ids based on specified legislative periods --- ##
def filter_legislature(periods, votes):
    '''
    Extract ids from votes dataset for affairs & councillors during specified legislative periods.
    input: 
        - periods: list of legislative periods (e.g. [49, 50])
        - votes: DataFrame containing at least 'date_clean', 'id', and 'elanId'
    output: 
        - set of affair ids 
        - set of councillor elan ids
    raises:
         ValueError if any period is not in mapping or if no vote data is found
    '''
    # maps periods to start/end date
    leg_mapping = pd.read_excel(MAP_DIR / 'legislative_period_mapping.xlsx')

    available_periods = set(leg_mapping['leg_period'])

    # Check 1: error if period not in mapping
    missing = [p for p in periods if p not in available_periods]
    if missing:
        raise ValueError(f"Legislative periods not found in mapping: {missing}")

    # compute date thresholds
    min_date = pd.to_datetime(
        leg_mapping[leg_mapping['leg_period'] == min(periods)]['start'].iloc[0]
    )
    max_date = pd.to_datetime(
        leg_mapping[leg_mapping['leg_period'] == max(periods)]['end'].iloc[0]
    )

    # filter votes dataset according to thresholds
    votes_filtered = votes[
        (votes['date_clean'] >= min_date) & (votes['date_clean'] <= max_date)
    ]

    # Check 2: error if no data
    if votes_filtered.empty:
        raise ValueError(
            f"No vote data found for legislative periods {periods} "
            f"(date range: {min_date.date()} to {max_date.date()})"
        )

    return set(votes_filtered['id']), set(votes_filtered['elanId'])

## --- filter votes, affairs and councillors based on ids using above function --- ##
def filter_all(votes, affairs, councillors, periods):
    '''
    Filters votes, affairs, and councillors to those active during the given legislative periods.
    Ensures consistency: only affairs and councillors with matching vote data are kept.
    '''
    # Ensure consistent integer dtypes
    votes['id'] = votes['id'].astype(int)
    affairs['id'] = affairs['id'].astype(int)
    votes['elanId'] = votes['elanId'].astype(int)
    councillors['elanId'] = councillors['elanId'].astype(int)

    # Extract relevant IDs from legislative period
    leg_affair_ids, leg_councillor_ids = filter_legislature(periods, votes)

    # Step 1: initial filter on votes
    filtered_votes = votes[
        votes['id'].isin(leg_affair_ids) &
        votes['elanId'].isin(leg_councillor_ids)
    ]

    # Step 2: only keep matching affair IDs
    common_affair_ids = set(filtered_votes['id']).intersection(set(affairs['id']))
    filtered_votes = filtered_votes[filtered_votes['id'].isin(common_affair_ids)]
    filtered_affairs = affairs[affairs['id'].isin(common_affair_ids)]

    # Step 3: only keep councillors who are actually in filtered_votes
    common_councillor_ids = set(filtered_votes['elanId']).intersection(set(councillors['elanId']))
    filtered_votes = filtered_votes[filtered_votes['elanId'].isin(common_councillor_ids)]
    filtered_councillors = councillors[councillors['elanId'].isin(common_councillor_ids)]

    return filtered_votes, filtered_affairs, filtered_councillors


## --- wrapper function to load data for given period(s) --- ##
def load_data(periods):
    votes = pd.read_parquet(CLEAN_DIR / 'votes.parquet')
    councillors = pd.read_excel(CLEAN_DIR / 'councillors.xlsx')
    affairs = pd.read_parquet(CLEAN_DIR / 'affairs.parquet')
    return filter_all(votes, affairs, councillors, periods)