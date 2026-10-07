import pandas as pd
import pytest

from loadcast.features import (
    MIN_LAG_HOURS,
    calendar_features,
    check_information_set,
    target_hours,
)


def test_calendar_uses_local_time_and_flags_holidays():
    index = pd.date_range("2024-01-05 22:00", periods=4, freq="1h", tz="UTC")
    cal = calendar_features(index, "ES", "Europe/Madrid")
    # 23:00 UTC on Jan 5 is already Jan 6 (Epiphany, national holiday) in Madrid.
    assert cal["hour"].tolist() == [23, 0, 1, 2]
    assert cal["is_holiday"].tolist() == [0, 1, 1, 1]


def test_bridge_day_between_holiday_and_weekend():
    # 2023-12-08 (Fri) is a holiday in Spain -> no bridge; 2022-10-31 (Mon) sits
    # between a weekend and All Saints' Day (Tue 1 Nov) -> bridge.
    index = pd.DatetimeIndex(["2022-10-31 12:00"], tz="UTC")
    assert calendar_features(index, "ES", "Europe/Madrid")["is_bridge"].item() == 1


def test_information_set_guard():
    check_information_set(issue_hour_utc=9, horizon_hours=24)  # max lead 39h
    with pytest.raises(ValueError):
        check_information_set(issue_hour_utc=0, horizon_hours=MIN_LAG_HOURS)


def test_target_hours_covers_whole_days():
    days = pd.DatetimeIndex(["2024-03-01", "2024-03-02"], tz="UTC")
    hours = target_hours(days, 24)
    assert len(hours) == 48 and hours[0] == days[0] and hours[-1].hour == 23
