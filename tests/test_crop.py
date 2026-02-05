from app.crop_math import compute_crop_box


def test_compute_crop_box_centers_face() -> None:
    box = compute_crop_box(
        face_center=(500.0, 300.0),
        face_size=(200.0, 160.0),
        image_size=(1000, 1000),
        roi_expand=2.0,
        face_y_ratio=3.0 / 7.0,
    )
    assert box == (300, 129, 700, 529)


def test_compute_crop_box_clamps_bounds() -> None:
    box = compute_crop_box(
        face_center=(50.0, 50.0),
        face_size=(100.0, 100.0),
        image_size=(400, 400),
        roi_expand=2.0,
        face_y_ratio=3.0 / 7.0,
    )
    left, top, right, bottom = box
    assert left == 0
    assert top == 0
    assert right <= 400
    assert bottom <= 400


def test_compute_crop_box_caps_size() -> None:
    box = compute_crop_box(
        face_center=(300.0, 300.0),
        face_size=(400.0, 400.0),
        image_size=(500, 500),
        roi_expand=2.0,
        face_y_ratio=3.0 / 7.0,
    )
    left, top, right, bottom = box
    assert right == 500
    assert bottom == 500
