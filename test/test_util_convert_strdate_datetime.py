def test_str_datetime_conversion():
    from stock_vcpscreener.vcp_util.util import convert_date_str_to_datetime
    import datetime

    testdate = datetime.datetime(2021, 8, 30)
    testconv = convert_date_str_to_datetime("2021-08-30")

    assert testconv.date() == testdate.date()


if __name__ == "__main__":
    test_str_datetime_conversion()
