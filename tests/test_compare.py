from car_seal.comparison.compare import calculate_iou


def test_calculate_iou():
    box1 = ["green", 200, 200, 300, 300]
    box2 = ["green", 400, 300, 200, 300]
    iou = calculate_iou(box1, box2)
    iou_opposite = calculate_iou(box2, box1)
    assert iou == iou_opposite
    assert iou == 2.0 / 13.0

    box1 = ["green", 200, 200, 300, 300]
    box2 = ["green", 600, 300, 200, 300]
    iou = calculate_iou(box1, box2)
    iou_opposite = calculate_iou(box2, box1)
    assert iou == iou_opposite
    assert iou == 0
