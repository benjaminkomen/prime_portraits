import time
import warnings
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import skimage.io
import skimage.transform
from PIL import Image, ImageFont, ImageDraw
from sklearn.cluster import KMeans


def get_k_means(image):
    point_cloud = np.reshape(image, (-1, 3))
    k_means = KMeans(n_clusters=10).fit(point_cloud)
    return k_means


def create_numbered_image(image, k_means):
    """
    Turns an RGB image into an image with the labels of clusters of the supplied k_means classifier.
    """
    # apply noise
    st_dev = np.std(image)
    random_noise = np.random.random_sample(size=image.shape) * (st_dev / 3)
    image = image + random_noise

    orig_shape = image.shape
    image = np.reshape(image, (-1, 3))

    numbered_image = k_means.predict(image)
    numbered_image = np.reshape(numbered_image, orig_shape[:2])

    # make sure the end is uneven
    if numbered_image[-1, -1] % 2 == 0:
        numbered_image[-1, -1] += 1

    return numbered_image


def numbered_image_to_normal_image(numbered_image, k_means):
    """
    Turns an image with only values between 0 and 9 into a colored image by using the cluster centers
    of the supplied k_means classifier.
    """
    shape = (numbered_image.shape[0], numbered_image.shape[1], 3)
    image = np.zeros(shape)
    for label, color in zip(range(10), k_means.cluster_centers_):
        image[numbered_image == label] = color
    return image


def load_and_resize_image(input_file_name, resize_factor=18):
    image = skimage.io.imread(input_file_name)
    image = image.astype(np.float64)
    image = image / 255.

    old_shape = image.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resized_image = skimage.transform.resize(image, (old_shape[0] // resize_factor, old_shape[1] // resize_factor))

    return resized_image


def image_to_number(numbered_image):
    to_be_number = numbered_image.reshape(-1)
    as_string = ''.join([str(int(x)) for x in to_be_number])
    as_int = int(as_string)
    return as_int, as_string


def show_and_save_image(image, n_image, input_file_name, fontsize=16):
    old_shape = image.shape
    resized_image = scipy.misc.imresize(image, (old_shape[0] * fontsize, old_shape[1] * fontsize), interp='nearest')
    img = Image.fromarray(resized_image).convert("RGBA")
    txt = Image.new('RGBA', img.size, (255, 255, 255, 0))

    draw = ImageDraw.Draw(txt)
    font = ImageFont.truetype("pirulen rg.ttf", fontsize)
    for y_i, to_type in enumerate(n_image):
        for x_i, letter in enumerate(to_type):
            x_pos = x_i * fontsize + 1
            y_pos = y_i * fontsize
            if letter == 1:
                x_pos += 4
            draw.text((x_pos, y_pos), str(letter), (255, 255, 255, 128), font=font)
    img = Image.alpha_composite(img, txt)

    img.save(input_file_name)
    plt.figure(figsize=(20, 20))
    plt.imshow(img)
    plt.show()


def result_filename(input_file_name):
    return input_file_name.split('.')[0] + "-prime.png"


def is_good_prime_portrait(n_image):
    integer, string = image_to_number(n_image)
    if is_probable_prime(integer):
        return integer, string, n_image
    else:
        return None


def print_result(string, n_image):
    print("Found a result: " + "-" * 100)
    print(string)
    print("Represented as portrait:" + "-" * 100)
    for line in n_image:
        print(''.join([str(x) for x in line]))


def multi_threaded_prime_generator(resized_image, k_means, input_file_name, threads=4, log_process=True):
    image_generator = (create_numbered_image(resized_image, k_means) for _ in range(1000000))
    start = time.time()
    with Pool(threads) as pool:
        results = pool.imap_unordered(is_good_prime_portrait, image_generator)
        total_results = 0

        for result in results:
            total_results += 1

            # Possibly log time spend searching this prime number
            if log_process and total_results % 30 == 0:
                elapsed = time.time()
                elapsed = elapsed - start
                print("Seconds spent in (function name) is {} time per result: {}".format(str(elapsed),
                                                                                          str(elapsed / total_results)))

            if result is not None:
                # Found a prime number, print it and save it!
                integer, string, n_image = result
                print_result(string, n_image)
                normal_image = numbered_image_to_normal_image(n_image, k_means)
                plt.imshow(normal_image)
                plt.show()
                show_and_save_image(normal_image, n_image, result_filename(input_file_name))
                break


def search_prime_portrait(input_file_name, resize_factor=16, log_process=True, threads=4):
    resized_image = load_and_resize_image(input_file_name, resize_factor=resize_factor)
    print("Working with size " + str(resized_image.shape))
    k_means = get_k_means(resized_image)
    multi_threaded_prime_generator(resized_image, k_means, input_file_name, log_process=log_process, threads=threads)


folder = 'input_pictures/'

filenames = [
    folder + 'tas.png',
]
filename = filenames[0]

for filename in filenames:
    search_prime_portrait(filename, resize_factor=20, log_process=False, threads=4)
