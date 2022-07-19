from os.path import realpath
from os.path import dirname
from os.path import join
from numpypurennmnist.load import MnistDataloader

def main():

    ROOT_DIR = realpath(dirname(__file__))

    training_images_filepath = join(ROOT_DIR, 'data', 'train-images-idx3-ubyte')
    training_labels_filepath = join(ROOT_DIR, 'data', 'train-labels-idx1-ubyte')
    test_images_filepath = join(ROOT_DIR, 'data', 't10k-images-idx3-ubyte')
    test_labels_filepath = join(ROOT_DIR, 'data', 't10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)

    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    return None

if __name__ == '__main__':
    main()



