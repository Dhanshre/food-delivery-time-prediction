import numpy as np
import pandas as pd


def change_column_names(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.rename(str.lower, axis=1)
        .rename(
            {
                "delivery_person_id": "rider_id",
                "delivery_person_age": "age",
                "delivery_person_ratings": "ratings",
                "delivery_location_latitude": "delivery_latitude",
                "delivery_location_longitude": "delivery_longitude",
                "time_orderd": "order_time",
                "time_order_picked": "order_picked_time",
                "weatherconditions": "weather",
                "road_traffic_density": "traffic",
                "city": "city_type",
                "time_taken(min)": "time_taken",
            },
            axis=1,
        )
    )


def time_of_day(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=[0, 6, 12, 17, 20, 24],
        right=True,
        labels=[
            "after_midnight",
            "morning",
            "afternoon",
            "evening",
            "night",
        ],
    )


def data_cleaning(data: pd.DataFrame) -> pd.DataFrame:
    data = data.replace("NaN ", np.nan).copy()

    age_numeric = pd.to_numeric(data["age"], errors="coerce")
    ratings_numeric = pd.to_numeric(data["ratings"], errors="coerce")

    invalid_rows = (age_numeric < 18) | (ratings_numeric == 6)
    data = data.loc[~invalid_rows].copy()

    cleaned_data = (
        data.drop(columns=["id"], errors="ignore")
        .assign(
            city_name=lambda x: x["rider_id"].str.split("RES").str.get(0),

            age=lambda x: pd.to_numeric(
                x["age"],
                errors="coerce",
            ),

            ratings=lambda x: pd.to_numeric(
                x["ratings"],
                errors="coerce",
            ),

            restaurant_latitude=lambda x: pd.to_numeric(
                x["restaurant_latitude"],
                errors="coerce",
            ).abs(),

            restaurant_longitude=lambda x: pd.to_numeric(
                x["restaurant_longitude"],
                errors="coerce",
            ).abs(),

            delivery_latitude=lambda x: pd.to_numeric(
                x["delivery_latitude"],
                errors="coerce",
            ).abs(),

            delivery_longitude=lambda x: pd.to_numeric(
                x["delivery_longitude"],
                errors="coerce",
            ).abs(),

            order_date=lambda x: pd.to_datetime(
                x["order_date"],
                dayfirst=True,
                errors="coerce",
            ),

            order_day=lambda x: x["order_date"].dt.day,

            order_month=lambda x: x["order_date"].dt.month,

            order_day_of_week=lambda x: (
                x["order_date"].dt.day_name().str.lower()
            ),

            is_weekend=lambda x: (
                x["order_date"]
                .dt.day_name()
                .isin(["Saturday", "Sunday"])
                .astype(int)
            ),

            order_time=lambda x: pd.to_datetime(
                x["order_time"],
                format="mixed",
                errors="coerce",
            ),

            order_picked_time=lambda x: pd.to_datetime(
                x["order_picked_time"],
                format="mixed",
                errors="coerce",
            ),

            pickup_time_minutes=lambda x: (
                (
                    x["order_picked_time"]
                    - x["order_time"]
                ).dt.seconds
                / 60
            ),

            order_time_hour=lambda x: x["order_time"].dt.hour,

            order_time_of_day=lambda x: time_of_day(
                x["order_time_hour"]
            ),

            weather=lambda x: (
                x["weather"]
                .str.replace(
                    "conditions ",
                    "",
                    regex=False,
                )
                .str.lower()
                .replace("nan", np.nan)
            ),

            traffic=lambda x: (
                x["traffic"]
                .str.rstrip()
                .str.lower()
            ),

            type_of_order=lambda x: (
                x["type_of_order"]
                .str.rstrip()
                .str.lower()
            ),

            type_of_vehicle=lambda x: (
                x["type_of_vehicle"]
                .str.rstrip()
                .str.lower()
            ),

            festival=lambda x: (
                x["festival"]
                .str.rstrip()
                .str.lower()
            ),

            city_type=lambda x: (
                x["city_type"]
                .str.rstrip()
                .str.lower()
            ),

            multiple_deliveries=lambda x: pd.to_numeric(
                x["multiple_deliveries"],
                errors="coerce",
            ),
        )
        .drop(
            columns=[
                "order_time",
                "order_picked_time",
            ]
        )
    )

    # The target exists during training, but not during API prediction.
    if "time_taken" in cleaned_data.columns:
        cleaned_data["time_taken"] = pd.to_numeric(
            cleaned_data["time_taken"]
            .astype(str)
            .str.replace(
                "(min) ",
                "",
                regex=False,
            ),
            errors="coerce",
        )

    return cleaned_data


def clean_lat_long(
    data: pd.DataFrame,
    threshold: float = 1,
) -> pd.DataFrame:
    location_columns = [
        "restaurant_latitude",
        "restaurant_longitude",
        "delivery_latitude",
        "delivery_longitude",
    ]

    return data.assign(
        **{
            column: np.where(
                data[column] < threshold,
                np.nan,
                data[column].values,
            )
            for column in location_columns
        }
    )


def extract_datetime_features(series: pd.Series) -> pd.DataFrame:
    date_col = pd.to_datetime(
        series,
        dayfirst=True,
        errors="coerce",
    )

    return pd.DataFrame(
        {
            "day": date_col.dt.day,
            "month": date_col.dt.month,
            "year": date_col.dt.year,
            "day_of_week": date_col.dt.day_name(),
            "is_weekend": (
                date_col.dt.day_name()
                .isin(["Saturday", "Sunday"])
                .astype(int)
            ),
        }
    )


def calculate_haversine_distance(
    data: pd.DataFrame,
) -> pd.DataFrame:
    lat1 = data["restaurant_latitude"]
    lon1 = data["restaurant_longitude"]
    lat2 = data["delivery_latitude"]
    lon2 = data["delivery_longitude"]

    lon1, lat1, lon2, lat2 = map(
        np.radians,
        [lon1, lat1, lon2, lat2],
    )

    longitude_difference = lon2 - lon1
    latitude_difference = lat2 - lat1

    a = (
        np.sin(latitude_difference / 2.0) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(longitude_difference / 2.0) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c

    return data.assign(distance=distance)


def create_distance_type(
    data: pd.DataFrame,
) -> pd.DataFrame:
    return data.assign(
        distance_type=pd.cut(
            data["distance"],
            bins=[0, 5, 10, 15, 25],
            right=False,
            labels=[
                "short",
                "medium",
                "long",
                "very_long",
            ],
        )
    )


def perform_data_cleaning(
    data: pd.DataFrame,
    saved_data_path: str | None = None,
) -> pd.DataFrame:
    cleaned_data = (
        data.pipe(change_column_names)
        .pipe(data_cleaning)
        .pipe(clean_lat_long)
        .pipe(calculate_haversine_distance)
        .pipe(create_distance_type)
    )

    if saved_data_path:
        cleaned_data.to_csv(
            saved_data_path,
            index=False,
        )

    return cleaned_data


if __name__ == "__main__":
    data_path = "swiggy.csv"

    dataframe = pd.read_csv(data_path)
    print("Swiggy data loaded successfully")

    perform_data_cleaning(
        dataframe,
        saved_data_path="swiggy_cleaned.csv",
    )
