import requests
import pandas as pd

eu_country_codes = [
    "AT",  # Austria
    "BE",  # Belgium
    "BG",  # Bulgaria
    "HR",  # Croatia
    "CY",  # Cyprus
    "CZ",  # Czech Republic
    "DK",  # Denmark
    "EE",  # Estonia
    "FI",  # Finland
    "FR",  # France
    "DE",  # Germany
    "GR",  # Greece
    "HU",  # Hungary
    "IE",  # Ireland
    "IT",  # Italy
    "LV",  # Latvia
    "LT",  # Lithuania
    "LU",  # Luxembourg
    "MT",  # Malta
    "NL",  # Netherlands
    "PL",  # Poland
    "PT",  # Portugal
    "RO",  # Romania
    "SK",  # Slovakia
    "SI",  # Slovenia
    "ES",  # Spain
    "SE",  # Sweden
]

four_countries = [
    "LT",  # Lithuania
    "SK",  # Slovakia
    "DE",  # Germany
    "GR",  # Greece
]

def main():
    fetch_landmarks_and_save(["BG"], 'bg_landmarks.csv')


def fetch_landmarks_and_save(country_code_list, csv_name):
    all_landmarks = fetch_landmarks_by_country_list(country_code_list)
    print(len(all_landmarks))
    df = pd.DataFrame(all_landmarks)
    # Rename the id column to landmark_id
    df = df.rename(columns={'id': 'landmark_id'})
    # Select only the landmark_id column
    df_landmark_ids = df[['landmark_id']]
    # Save the DataFrame to a CSV file
    df_landmark_ids.to_csv(csv_name, sep='\t', index=False)


def fetch_landmarks_by_country_list(codes):
    """
    Fetches landmark IDs for a list of country codes.

    Args:
    - codes (list): List of two-letter country codes.

    Returns:
    - list: Combined list of landmark IDs.
    """
    all_landmarks = []

    for country_code in codes:
        country_url = f"https://storage.googleapis.com/gld-v2/data/train/country/{country_code}.json"
        response = requests.get(country_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch data for country {country_code}")

        landmarks = response.json()
        all_landmarks.extend(landmarks)

    return all_landmarks


if __name__ == "__main__":
    main()