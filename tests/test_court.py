from pbos import court


def test_dimensions():
    assert (court.WIDTH_FT, court.LENGTH_FT) == (20.0, 44.0)
    assert (court.NEAR_KITCHEN_Y, court.FAR_KITCHEN_Y) == (15.0, 29.0)


def test_twelve_keypoints_on_painted_lines():
    assert len(court.KEYPOINTS) == 12
    for x, y in court.KEYPOINTS.values():
        assert x in (0.0, 10.0, 20.0)
        assert y in (0.0, 15.0, 29.0, 44.0)


def test_zones():
    assert court.in_kitchen(10, 22)
    assert court.in_kitchen(0, 15)  # the line is part of the kitchen
    assert not court.in_kitchen(10, 14.9)
    assert court.in_court(10, -3, end_margin=4)
    assert not court.in_court(25, 10, side_margin=4)
