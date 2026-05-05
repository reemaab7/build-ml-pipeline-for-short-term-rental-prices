import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):
    expected_columns = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]
    these_columns = data.columns.tolist()
    assert set(expected_columns) == set(these_columns)


def test_neighborhood_names(data):
    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
    neigh = set(data["neighbourhood_group"].unique())
    assert set(known_names) == neigh


def test_proper_boundaries(data: pd.DataFrame):
    idx = data["longitude"].between(-74.25, -73.50) & data["latitude"].between(40.5, 41.2)
    assert idx.all()


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    dist1 = data["neighbourhood_group"].value_counts().sort_index()
    dist2 = ref_data["neighbourhood_group"].value_counts().sort_index()
    assert scipy.stats.entropy(dist1, dist2) < kl_threshold


def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000


def test_price_range(data, min_price, max_price):
    assert data["price"].between(min_price, max_price).all()
