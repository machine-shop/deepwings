import numpy as np
from numpy.testing import assert_array_equal
import scipy.ndimage as ndi
import pytest
from skimage.measure import regionprops
from skimage.transform import rotate

from method_features_extraction import image_processing as ip


@pytest.fixture(scope="module")
def gray_image():
    ones = np.ones((60, 100))
    ones[:, 0] = 30
    faded = np.cumsum(ones, axis=1)

    faded[30:40, 50:60] = 30
    faded[30:40, 10:20] = 25
    faded[10:12, 10:80] = 25
    faded[40:50, 80:90] = 100

    return faded


@pytest.fixture(scope="module")
def binary_image():
    ones = np.ones((60, 100))
    ones[10:13, :] = 0
    ones[:, 50:55] = 0
    ones[30:40, 10:20] = 0

    return ones


@pytest.fixture(scope="module")
def image_label():
    labels = np.zeros((60, 100), dtype=int)
    # to be discared, area_region < area_tot*1/1600
    # here area_region < 6000/1600 = 3.75

    # small regions
    labels[10, 10] = 233
    labels[20, 10] = 223

    # normal regions
    labels[10:20, 50:60] = 32
    labels[30:60, 50:80] = 8

    return labels


@pytest.fixture(scope="module")
def fake_shape():
    shape = np.zeros((5, 5))
    shape[1:, 4] = 1
    shape[2:4, 2] = 1
    shape[[1, 4], 3] = 1

    return shape


@pytest.fixture()
def rs_img_label():
    img_label = np.zeros((60, 100), dtype=int)

    # 1st_med (central cell)
    img_label[30:45, 40:60] = 1
    # 1st sub
    img_label[23:30, 40:55] = 2
    # 2nd sub
    img_label[23:30, 55:65] = 3
    # 3rd sub
    img_label[23:30, 65:75] = 4
    # marg
    img_label[15:23, 50:80] = 5
    # 2nd med
    img_label[30:50, 60:70] = 6
    # 2nd cub
    img_label[45:55, 40:60] = 7

    return img_label


def test_block_binarization(gray_image, step=1, side_block=5):
    expected = np.ones((60, 100))

    expected[30:40, 50:60] = 0
    expected[30:40, 10:20] = 0
    expected[10:12, 10:80] = 0
    expected[40:50, 80:90] = 0

    binary = ip.block_binarization(gray_image, step=step,
                                   side_block=side_block)

    subset_expected = expected[side_block: -side_block,
                               side_block: -side_block]

    subset_binary = binary[side_block: -side_block,
                           side_block: -side_block]

    assert_array_equal(subset_binary, subset_expected)


def test_clear_binary(binary_image):
    expected = np.ones((60, 100))
    expected[10:13, :] = 0
    expected[:, 50:55] = 0

    cleared = ip.clear_binary(binary_image)

    assert_array_equal(cleared, expected)


def test_filter_and_label(image_label):
    expected = np.zeros((60, 100), dtype=int)

    # normal regions
    expected[10:20, 50:60] = 2
    expected[30:60, 50:80] = 1

    filtered, counter = ip.filter_and_label(image_label)
    print(f'Detected regions : {counter}/2')

    assert_array_equal(filtered, expected)


def test_moore_neighborhood():
    expected = np.array([[10, 9], [9, 9], [9, 10], [9, 11],
                         [10, 11], [11, 11], [11, 10], [11, 9]])
    boundary = ip.moore_neighborhood([10, 10], [10, 9])
    assert_array_equal(boundary, expected)


def test_boundary_tracing(fake_shape):
    expected1 = np.array([[1, 3], [1, 4], [2, 4], [3, 4], [4, 4],
                          [4, 3], [3, 4], [2, 4]])
    expected2 = np.array([[1, 3], [1, 4], [2, 4], [3, 4], [4, 4],
                          [4, 3], [3, 2], [2, 2]])
    labels1, _ = ndi.label(fake_shape,
                           structure=ndi.generate_binary_structure(2, 1))
    labels2, _ = ndi.label(fake_shape,
                           structure=ndi.generate_binary_structure(2, 2))

    regions1 = regionprops(labels1)
    regions2 = regionprops(labels2)

    boundary1 = ip.boundary_tracing(regions1[0])
    boundary2 = ip.boundary_tracing(regions2[0])

    assert_array_equal(boundary1, expected1)
    assert_array_equal(boundary2, expected2)


def test_symetric_list():
    expected = np.array([0, 1, -1, 2, -2])
    sym = ip.symetric_list(5)
    assert_array_equal(sym, expected)


def test_rs_filter_area(rs_img_label):

    # Large region
    rs_img_label[3:33, 2:27] = 10

    # Small region
    rs_img_label[50:52, 2:4] = 11

    rs = ip.region_sorter(rs_img_label, "test")
    rs.filter_area()
    rs.update_img_label()

    assert rs.labels == [1, 2, 3, 4, 5, 6, 7]


def test_rs_filter_weird_shapes(rs_img_label):

    # Weird region
    rs_img_label[10:50, 3:5] = 10
    rs_img_label[1:10, 3:10] = 10
    rs_img_label[50:57, 3:15] = 10

    rs = ip.region_sorter(rs_img_label, 'test')
    rs.filter_weird_shapes()

    assert rs.labels == [1, 2, 3, 4, 5, 6, 7]


def test_rs_filter_neighbors(rs_img_label):

    # Isolated regions
    rs_img_label[10:30, 4: 20] = 10
    rs_img_label[40:50, 4: 20] = 11

    # Regions with only one neighbor
    rs_img_label[33:40, 30:40] = 12
    rs_img_label[15:23, 80:90] = 13
    rs_img_label[35:50, 70:80] = 14

    rs = ip.region_sorter(rs_img_label, 'test')
    rs.find_neighbors()
    rs.filter_neighbors()

    assert rs.labels == [1, 2, 3, 4, 5, 6, 7]


def test_rs_central_cell(rs_img_label):

    # Long region with multiple neighbors
    rs_img_label[20:, 20:23] = 20
    for i in range(15):
        rs_img_label[20+2*i:22+2*i, 15:20] = 20 + i

    # Region with multiple long neighbors
    for i in range(8):
        rs_img_label[30+i:32+i, 70:] = 10 + i

    rs = ip.region_sorter(rs_img_label, 'test')
    rs.find_neighbors()
    rs.compute_eccentricities()
    rs.find_central_cell()
    assert rs.label_center == 1


def test_rs_compute_angles(rs_img_label):
    expected_angles = np.array([0, -102.8, -47.7, -28.8, -51, 9.5, 90])
    center = (49, 26)  # x, y
    rotation = -10

    rotated = np.array(rotate(rs_img_label, rotation, center=center, order=0,
                              mode='edge', preserve_range=True),
                       dtype=int)

    rs = ip.region_sorter(rotated, 'test')
    rs.label_center = 1
    rs.compute_angles()
    angles_adjusted = np.array(rs.angles)
    print(angles_adjusted)

    diff_adj = np.abs(angles_adjusted - expected_angles)

    rs.compute_angles(adjusted=False)
    angles_no_adjusted = np.array(rs.angles)
    print(angles_no_adjusted)
    adjustment = np.ones((7))*rotation
    adjustment[0] = 0

    diff_no_adj = np.abs(angles_no_adjusted + adjustment - expected_angles)

    assert np.max(diff_adj) < 1
    assert np.max(diff_no_adj) < 1

    assert angles_adjusted[0] == 0.0
    assert angles_no_adjusted[0] == 0.0


def test_rs_filter_angles(rs_img_label):

    # Cells to discard
    rs_img_label[15:23, 40:50] = 10
    rs_img_label[23:30, 30:40] = 11
    rs_img_label[50:, 60:80] = 12
    rs_img_label[40:50, 70:80] = 13
    rs_img_label[23:30, 75:90] = 14
    rs_img_label[30:45, 25:30] = 15

    # Adjust central cell
    rs_img_label[30:45, 30:40] = 1

    # Adjust marg cell
    rs_img_label[15:23, 50:60] = 0

    rs = ip.region_sorter(rs_img_label, 'test')
    rs.label_center = 1
    rs.find_neighbors()
    rs.compute_angles()
    rs.filter_angles()

    assert rs.labels == [1, 2, 3, 4, 5, 6, 7]


def test_rs_filter_annoying_cell(rs_img_label):
    # Annoying cell
    rs_img_label[30:40, 70:80] = 10

    rs = ip.region_sorter(rs_img_label, 'test')
    rs.label_center = 1
    rs.find_neighbors()
    rs.compute_angles()
    rs.filter_annoying_cell()

    assert rs.labels == [1, 2, 3, 4, 5, 6, 7]


def test_rs_identify_regions(rs_img_label):
    rs = ip.region_sorter(rs_img_label, 'test')
    rs.label_center = 1
    rs.find_neighbors()
    rs.compute_angles()
    rs.identify_regions()

    assert rs.cells == ['1st_med', '1st_sub', '2nd_sub', '3rd_sub',
                        'marg', '2nd_med', '2nd_cub']
