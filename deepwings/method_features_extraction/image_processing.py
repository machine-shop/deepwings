import cmath
from cmath import exp, polar, pi
import numpy as np
import time

from itertools import combinations

from scipy import ndimage as ndi
from scipy.spatial import distance
from skimage.color import label2rgb
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import reconstruction, watershed
from skimage.segmentation import clear_border


def sub_block_binarization(image, block_size=(100, 100), offset=0):
    """Divide the image into blocks, and binarize each one of them with
    Otsu binarization

    Arguments
    ---------
    image : 2D array
        Grayscale image to binarize
    block_size : tuple
        Size of each block
    offset : int
        Number of pixels between the first column of blocks and the left side
        of the picture. Number of pixels between the first row of blocks
        and the upper bound of the picture.

    Returns
    -------
    binary : 2D array
        Binary image
    """

    width = image.shape[1]
    height = image.shape[0]
    bw = block_size[1]
    bh = block_size[0]
    nw = width // bw
    nh = height // bh
    binary = np.zeros((image.shape[0], image.shape[1]))

    if (offset):
        nw -= 1
        nh -= 1

    for i in range(nw):
        for j in range(nh):
            x_start = offset + i * bw
            y_start = offset + j * bh
            focus = image[y_start: y_start + bh, x_start: x_start + bw]
            try:
                thresh = threshold_otsu(focus, nbins=60)
                focus = focus > thresh
            except ValueError:
                focus = np.zeros((bh, bw))
            binary[y_start: y_start + bh, x_start: x_start + bw] = focus

    return binary


def block_binarization(image, step=20, side_block=100):
    """Superpose different block-binarized pictures with different offsets into
    one global binarized picture

    Arguments
    ---------
    image : 2D array
        Grayscale image to binarize
    step : int
        Step between each offset of the sub block binarized images
    side_block : int
        Side of the blocks in pixels

    Returns
    -------
    binary : 2D array
        block binarized image
    """

    total = sub_block_binarization(image, block_size=(side_block, side_block),
                                   offset=0)
    offset = step
    while (offset % side_block != 0):
        total += sub_block_binarization(image, block_size=(side_block,
                                                           side_block),
                                        offset=offset)
        offset += step
    thresh = threshold_otsu(total, nbins=60)

    return total > thresh


def clear_binary(binary):
    """Remove pixels disconnected from to veins

    Arguments
    ---------
    binary : 2D array
        Block binarized image

    Returns
    -------
    reconstructed : 2D array
        Binarized image of veins cleared
    """

    seed = np.copy(binary)
    seed[:, 1:-1] = binary.max()
    mask = binary

    reconstructed = reconstruction(seed, mask, method='erosion')
    return reconstructed


def filter_and_label(img_label):
    """Filter regions by area and change labels of each filtered region
    to avoid labels > 255

    Parameters
    ----------
    img_label (array): Image label

    Retunrs
    -------
    re_labeled (array): Image label with filtered and re-labeled regions
    counter (int): Number of filtered regions
    """

    t = time.time()
    re_labeled = np.zeros(img_label.shape, dtype=int)
    regions = regionprops(img_label, coordinates='xy')
    counter = 0

    threshold = img_label.shape[0]*img_label.shape[1]*6.25e-4
    for region in regions:
        area = region.area
        if area > threshold:
            counter += 1
            coords = region.coords
            y = coords[:, 0]
            x = coords[:, 1]
            re_labeled[tuple([y, x])] = counter
    duration = round(time.time() - t, 4)
    print(f'# Filter and label lasted {duration}s')
    return re_labeled, counter


def moore_neighborhood(current, backtrack):  # y, x
    """Returns clockwise list of pixels from the moore neighborhood of current\
    pixel:
    The first element is the coordinates of the backtrack pixel.
    The following elements are the coordinates of the neighboring pixels in
    clockwise order.

    Parameters
    ----------
    current ([y, x]): Coordinates of the current pixel
    backtrack ([y, x]): Coordinates of the backtrack pixel

    Returns
    -------
    List of coordinates of the moore neighborood pixels, or 0 if the backtrack
    pixel is not a current pixel neighbor
    """

    operations = np.array([[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1],
                           [0, -1], [-1, -1]])
    neighbors = (current + operations).astype(int)

    for i, point in enumerate(neighbors):
        if np.all(point == backtrack):
            # we return the sorted neighborhood
            return np.concatenate((neighbors[i:], neighbors[:i]))
    return 0


def boundary_tracing(region):
    """Coordinates of the region's boundary. The region must not have isolated
    points.

    Parameters
    ----------
    region : obj
        Obtained with skimage.measure.regionprops()

    Returns
    -------
    boundary : 2D array
        List of coordinates of pixels in the boundary
        The first element is the most upper left pixel of the region.
        The following coordinates are in clockwise order.
    """

    # creating the binary image
    coords = region.coords
    maxs = np.amax(coords, axis=0)
    binary = np.zeros((maxs[0] + 2, maxs[1] + 2))
    x = coords[:, 1]
    y = coords[:, 0]
    binary[tuple([y, x])] = 1

    # initilization
    # starting point is the most upper left point
    idx_start = 0
    while True:  # asserting that the starting point is not isolated
        start = [y[idx_start], x[idx_start]]
        focus_start = binary[start[0]-1:start[0]+2, start[1]-1:start[1]+2]
        if np.sum(focus_start) > 1:
            break
        idx_start += 1

    # Determining backtrack pixel for the first element
    if (binary[start[0] + 1, start[1]] == 0 and
            binary[start[0]+1, start[1]-1] == 0):
        backtrack_start = [start[0]+1, start[1]]
    else:
        backtrack_start = [start[0], start[1] - 1]

    current = start
    backtrack = backtrack_start
    boundary = []
    counter = 0

    while True:
        neighbors_current = moore_neighborhood(current, backtrack)
        y = neighbors_current[:, 0]
        x = neighbors_current[:, 1]
        idx = np.argmax(binary[tuple([y, x])])
        boundary.append(current)
        backtrack = neighbors_current[idx-1]
        current = neighbors_current[idx]
        counter += 1

        if (np.all(current == start) and np.all(backtrack == backtrack_start)):
            break

    return np.array(boundary)


def symetric_list(n):
    """Returns a list of the n first elements of a symetric list
    Example:
        symetric_list(5) returns [0, 1, -1, 2, -2]
    """
    output = []
    for i in range(n):
        if i % 2 == 0:
            output.append(-i/2)
        else:
            output.append((i+1)/2)
    return np.array(output).astype(int)


def fourier_descriptors(boundary, n_descriptors):
    """Returns a list of the first complex Fourier descriptors of a boundary

    Parameters
    ----------
    boundary : 2D array
        List of coordinates of pixels part of the boundary
    n_descriptors : int
        Number of complex Fourier descriptors wanted

    Returns
    -------
    descriptors : 1D list
        List of the first (n_descriptors) complex Fourier descriptors of a
        boundary
    """
    y = boundary[:, 0]
    x = boundary[:, 1]
    complex_boundary = x + y*1j
    n = len(boundary)
    descriptors = []
    k_values = symetric_list(n_descriptors)
    for p in range(n_descriptors):
        sum_c = 0
        k = k_values[p]
        for i in range(n):
            sum_c += complex_boundary[i] * exp(-2*pi*1j*(i+1)*k/n)
        descriptors.append(round((sum_c/n).real, 3) +
                           round((sum_c/n).imag, 3)*1j)
    return descriptors


def normalize_descriptors(complex_descriptors):
    """Take a list of complex Fourier descriptors, discards descriptors 0 and 1
    and normalizes the other ones by the modulus of descriptor 1.
    Phases are discarded too.
    """
    mod_c1 = polar(complex_descriptors[1])[0]
    return ([round(polar(descriptor)[0]/mod_c1, 4)
             for descriptor in complex_descriptors[2:]])


def create_img_label(cleared_binary):
    """Create image label from binary picture

    Arguments
    ---------
    cleared_binary : 2D array
        Binary image cleared

    Returns
    -------
    img_label : 2D array
        Image label with cleared borders
    markers : 2D array
        Image label with relabelled regions
    distances : 2D array
        Distance map, the value of each pixel corresponds to its distance
        to the closest pixel of background
    labels : 2D array
        Image label after watershed, with black border
    """
    t = time.time()

    markers, _ = ndi.label(cleared_binary,
                           structure=ndi.generate_binary_structure(2, 1))
    markers, num_features = filter_and_label(markers)
    if num_features > 255:
        print("### Warning : more than 255 regions detected")
    distances = ndi.distance_transform_edt(cleared_binary)
    labels = watershed(-distances, markers)
    labels[0, :] = 0
    labels[:, -1] = 0
    labels[:, 0] = 0
    img_label = clear_border(labels)

    duration = round(time.time() - t, 4)
    print(f'# Creating img label lasted {duration}s')

    return img_label, markers, distances, labels


class region_sorter():
    """ Once we have the image label, we only keep cells of interest."""

    def __init__(self, img_label, file_name):
        self.file_name = file_name
        self.img_label = img_label
        self.area_tot = img_label.shape[0]*img_label.shape[1]
        self.height = img_label.shape[0]
        self.width = img_label.shape[1]
        self.regions = regionprops(img_label, coordinates='xy')
        self.labels = []
        self.areas = []
        self.centroids_x = []
        self.centroids_y = []

        # Lists initialized later
        self.angles = None
        self.neighbors = None
        self.cells = None
        self.eccentricities = None
        self.fd = None

        self.label_center = None
        self.center_x = None
        self.center_y = None
        self.valid_image = None

        for region in self.regions:
            self.labels.append(region.label)
            self.areas.append(region.area)
            self.centroids_y.append(region.centroid[0])
            self.centroids_x.append(region.centroid[1])

    def filter_area(self, area_min=1.656e-3, area_max=9.3e-2):
        # 9.3e-2
        t = time.time()

        thresh_min = self.area_tot*area_min
        thresh_max = self.area_tot*area_max

        to_delete = []
        for i in range(len(self.areas)):
            area = self.areas[i]
            if area < thresh_min or area > thresh_max:
                to_delete.append(i)
        self.delete(to_delete)
        duration = round(time.time() - t, 4)
        print(f'# Filter area lasted {duration}s')

    def filter_weird_shapes(self):
        to_delete = []

        for i in range(len(self.labels)):
            y = int(self.centroids_y[i])
            x = int(self.centroids_x[i])
            label = self.labels[i]
            if self.img_label[y, x] != label:
                to_delete.append(i)
        self.delete(to_delete)

    def delete(self, to_delete):
        to_delete = sorted(set(to_delete), reverse=True)

        for i in to_delete:
            del self.labels[i]
            del self.areas[i]
            del self.regions[i]
            del self.centroids_x[i]
            del self.centroids_y[i]
            if(self.neighbors):
                del self.neighbors[i]
            if(self.angles):
                del self.angles[i]
            if(self.eccentricities):
                del self.eccentricities[i]
            if(self.fd):
                del self.fd[i]

    def update_img_label(self):
        t = time.time()
        updated_img_label = np.zeros((self.img_label.shape[0],
                                      self.img_label.shape[1]))
        for region in self.regions:
            coords = region.coords
            y = coords[:, 0]
            x = coords[:, 1]
            updated_img_label[tuple([y, x])] = region.label
        self.img_label = updated_img_label
        duration = round(time.time() - t, 4)
        print(f'# Update image label lasted {duration}s')

    def find_neighbors(self, blockshape=(2, 2)):
        t = time.time()

        up = self.img_label[:-1, :]
        down = self.img_label[1:, :]
        left = self.img_label[:, :-1]
        right = self.img_label[:, 1:]

        # Vertical superposition
        vert_changes = np.logical_and(up != down,
                                      np.logical_and(up != 0, down != 0))
        pairs_v = np.array([up[vert_changes], down[vert_changes]])
        pairs_v = np.transpose(pairs_v)
        pairs_v = np.sort(pairs_v, axis=1)

        # Horizontal superposition
        hori_changes = np.logical_and(left != right,
                                      np.logical_and(left != 0, right != 0))
        pairs_h = np.array([left[hori_changes],
                            right[hori_changes]])
        pairs_h = np.transpose(pairs_h)
        pairs_h = np.sort(pairs_h, axis=1)

        # All pairs
        pairs_tot = np.concatenate((pairs_v, pairs_h), axis=0)
        pairs_tot = np.unique(pairs_tot, axis=0)

        all_neighbors = [[]] * len(self.labels)

        for pair in list(pairs_tot):
            idx0 = self.labels.index(pair[0])
            all_neighbors[idx0] = all_neighbors[idx0] + [pair[1]]
            idx1 = self.labels.index(pair[1])
            all_neighbors[idx1] = all_neighbors[idx1] + [pair[0]]

        all_neighbors = [list(set(neighbors)) for neighbors in all_neighbors]
        self.neighbors = all_neighbors

        duration = round(time.time() - t, 4)
        print(f'# Find neighbors lasted {duration}s')

    def filter_neighbors(self, neighbors_min=2):
        t = time.time()
        while True:
            to_delete = []
            for i in range(len(self.labels)):
                if (len(self.neighbors[i]) < neighbors_min):
                    to_delete.append(i)
            self.delete(to_delete)

            if len(to_delete) == 0:
                break

            new_neighbors = []
            for neighbors in self.neighbors:
                buffer = []
                for neighbor in neighbors:
                    if neighbor in self.labels:
                        buffer.append(neighbor)
                new_neighbors.append(buffer)

            self.neighbors = new_neighbors
        duration = round(time.time() - t, 4)
        print(f'# Filter neighbors lasted {duration}s')

    def update_center(self):
        self.center_x = int(np.mean(self.centroids_x))
        self.center_y = int(np.mean(self.centroids_y))

    def compute_eccentricities(self):
        self.eccentricities = ([round(region.eccentricity, 3)
                                for region in self.regions])

    def find_central_cell(self):
        t = time.time()

        count_exotic = np.zeros((len(self.labels),))
        for i, neighbors in enumerate(self.neighbors):
            for neighbor in neighbors:
                idx = self.labels.index(neighbor)
                if self.eccentricities[idx] > 0.95:
                    count_exotic[i] += 1
        nb_neighbors = np.array(
            [len(list_neighbors) for list_neighbors in self.neighbors])

        nb_valid_neighbors = nb_neighbors - count_exotic

        score = nb_valid_neighbors*np.array(self.areas)
        score[np.array(self.centroids_x) > 0.6*self.img_label.shape[1]] = 0
        score[np.array(self.eccentricities) > 0.975] = 0

        self.label_center = self.labels[np.argmax(score)]

        duration = round(time.time() - t, 4)
        print(f'# Find central cell lasted {duration}s')

    def compute_angles(self, adjusted=True):
        t = time.time()

        idx_center = self.labels.index(self.label_center)
        center_complex = complex(self.centroids_x[idx_center],
                                 self.centroids_y[idx_center])
        centroids_complex = []
        for i in range(len(self.labels)):
                centroids_complex.append(complex(self.centroids_x[i],
                                                 self.centroids_y[i]))
        vects_complex = np.array(centroids_complex) - center_complex
        if adjusted:
            orientation = self.regions[idx_center].orientation
            rotated = vects_complex*cmath.rect(1, orientation)
            angles = np.angle(rotated, deg=True)
        else:
            angles = np.angle(vects_complex, deg=True)
        angles[idx_center] = 0
        self.angles = [round(angle, 1) for angle in angles]

        duration = round(time.time() - t, 4)
        print(f'# Computing angles lasted {duration}s')

    def filter_angles(self):
        t = time.time()

        to_delete = []
        idx_center = self.labels.index(self.label_center)
        for i, label in enumerate(self.labels):
            if i != idx_center:
                angle = self.angles[i]
                if label in self.neighbors[idx_center]:
                    if (angle < -80 or angle > 150):  # -72  126
                        to_delete.append(i)
                else:
                    degree2 = False
                    for neighbor in self.neighbors[i]:
                        if neighbor in self.neighbors[idx_center]:
                            degree2 = True
                            if (angle < -40 or angle > 5):
                                to_delete.append(i)
                    if not degree2:
                        to_delete.append(i)
        self.delete(to_delete)

        duration = round(time.time() - t, 4)
        print(f'# Filter angles lasted {duration}s')

    def filter_annoying_cell(self):
        """ Sometimes, we have more than 2 cells not neighbors with central cell
        We want to discard those with biggest angles.
        """

        idx_center = self.labels.index(self.label_center)
        not_neighbors = []
        for i, label in enumerate(self.labels):
            if label not in self.neighbors[idx_center] and i != idx_center:
                not_neighbors.append([i, self.angles[i]])
        not_neighbors = np.array(not_neighbors)
        if len(not_neighbors) > 2:
            to_delete = not_neighbors[not_neighbors[:, 1].argsort()][2:, 0]
            self.delete(to_delete.astype(int))

    def identify_regions(self):
        t = time.time()

        self.valid_image = True
        nb_regions = len(self.labels)

        if (nb_regions not in [6, 7]):
            self.valid_image = False

        if(self.valid_image):
            self.cells = []
            buff = np.zeros(nb_regions)

            idx_center = self.labels.index(self.label_center)
            neighbors_center = self.neighbors[idx_center]
            pairs = []
            for i in range(nb_regions):
                data = [i, self.angles[i]]
                if i == idx_center:
                    buff[i] = nb_regions - 1
                else:
                    pairs.append(data)
            pairs = sorted(pairs, key=lambda pair: pair[1])

            found = False
            counter = 0
            for j in range(len(pairs)):
                idx = pairs[j][0]
                if self.labels[idx] not in neighbors_center and not found:
                    counter += 1
                    if not found:
                        found = True
                        buff[idx] = nb_regions - 2
                else:
                    if not found:
                        buff[idx] = j
                    if found:
                        buff[idx] = j - 1
            if counter > 2:
                self.valid = False

            names = ['1st_sub', '2nd_sub', '3rd_sub', '2nd_med', '2nd_cub',
                     'marg', '1st_med']
            if nb_regions == 6:
                del names[2]

            for i in range(nb_regions):
                self.cells.append(names[int(buff[i])])

        if not self.valid_image:
            self.cells = None

        duration = round(time.time() - t, 4)
        print(f'# Identifying regions lasted {duration}s')

    def plot(self, ax, img_gray=None, fontsize=7):
        t = time.time()

        self.update_img_label()

        # print('mixing images')
        if img_gray:
            background = label2rgb(self.img_label,
                                   image=img_gray,
                                   bg_label=0)
        else:
            background = self.img_label

        ax.imshow(background)
        ax.scatter(self.centroids_x, self.centroids_y, color='r')

        step = int(self.height*0.02)
        offset_x = int(self.width*0.015)
        for i in range(len(self.labels)):
            offset_y = 0
            ax.text(self.centroids_x[i] + offset_x,
                    self.centroids_y[i] + offset_y,
                    "l: " + str(self.labels[i]),
                    color='white',
                    fontsize=fontsize,
                    fontweight='bold')
            offset_y += step
            ax.text(self.centroids_x[i] + offset_x,
                    self.centroids_y[i] + offset_y,
                    "a: "+str(self.areas[i]),
                    color='white',
                    fontsize=fontsize,
                    fontweight='bold')

            if (self.angles):
                offset_y += step
                ax.text(self.centroids_x[i] + offset_x,
                        self.centroids_y[i] + offset_y,
                        "o: "+str(self.angles[i])+"°",
                        color='white',
                        fontsize=fontsize,
                        fontweight='bold')

            if self.cells and self.valid_image:
                offset_y += step
                ax.text(self.centroids_x[i] + offset_x,
                        self.centroids_y[i] + offset_y,
                        "c: "+str(self.cells[i]),
                        color='white',
                        fontsize=fontsize,
                        fontweight='bold')

            if self.eccentricities:
                offset_y += step
                ax.text(self.centroids_x[i] + offset_x,
                        self.centroids_y[i] + offset_y,
                        "e: "+str(self.eccentricities[i]),
                        color='white',
                        fontsize=fontsize,
                        fontweight='bold')

        if(self.center_x):
            ax.scatter(self.center_x, self.center_y, color='b')

        if(self.label_center):
            offset_y += step
            idx = self.labels.index(self.label_center)
            ax.text(self.centroids_x[idx] + offset_x,
                    self.centroids_y[idx] + offset_y,
                    "(central cell)",
                    color='white',
                    fontsize=fontsize,
                    fontweight='bold')

        if self.valid_image is not None:
            if self.valid_image:
                txt = 'Image valid'
                color = 'g'
            else:
                txt = 'Image not valid'
                color = 'r'
            ax.text(0, 70, txt, color=color, fontsize=2*fontsize,
                    fontweight='bold')

        duration = round(time.time() - t, 4)
        print(f'# Plotting lasted {duration}s')

    def compute_fd(self, n_descriptors):  # descriptors = total - d0 and d1
        t = time.time()
        if self.valid_image:
            self.fd = []
            for idx, label in enumerate(self.labels):
                boundary = boundary_tracing(self.regions[idx])
                descriptors = fourier_descriptors(boundary, n_descriptors+2)
                normalized = normalize_descriptors(descriptors)
                self.fd.append(normalized)

        duration = round(time.time() - t, 4)
        print(f'# Computing Fourier descriptors lasted {duration}s')

    def calculate_distance(self, pair):
        idx0 = self.cells.index(pair[0])
        idx1 = self.cells.index(pair[1])
        point0 = (self.centroids_y[idx0], self.centroids_x[idx0])
        point1 = (self.centroids_y[idx1], self.centroids_x[idx1])
        return round(distance.euclidean(point0, point1), 3)

    def extract_features(self, compute_ratios=False):
        t = time.time()

        if not self.valid_image:
            nb_cells, output = None, None
        else:
            nb_cells = len(self.labels)
            cells_ordered = ['marg', '1st_med', '2nd_med', '2nd_cub',
                             '1st_sub', '2nd_sub', '3rd_sub']
            if nb_cells == 6:
                cells_ordered = cells_ordered[:-1]

            output = [self.file_name]
            total_area = sum(self.areas)
            area = []
            eccentricity = []
            angles = []
            fourier_descriptors = []
            for cell in cells_ordered:  # to avoid redundancy
                idx = self.cells.index(cell)
                area += [round(self.areas[idx]/total_area, 3)]
                eccentricity += [self.eccentricities[idx]]
                fourier_descriptors += self.fd[idx]
                if self.angles[idx] != 0:
                    angles += [self.angles[idx]]

            output += (area[:-1] + eccentricity + angles + fourier_descriptors)

            if compute_ratios:
                comb_cells = combinations(cells_ordered, 2)
                distances = []
                for pair in list(comb_cells):
                    distances.append(self.calculate_distance(pair))

                comb_dist = combinations(range(len(distances)), 2)
                for pair_idx in comb_dist:
                    ratio = distances[pair_idx[0]] / distances[pair_idx[1]]
                    output += [round(ratio, 4)]

        duration = round(time.time() - t, 4)
        print(f'# Features extraction lasted {duration}s')

        return nb_cells, output
