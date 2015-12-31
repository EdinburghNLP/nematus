#!/usr/bin/python

import argparse
import logging
import os
import tarfile
import urllib2

TRAIN_DATA_URL = 'http://www.statmt.org/europarl/v7/fr-en.tgz'
VALID_DATA_URL = 'http://matrix.statmt.org/test_sets/newstest2011.tgz'

parser = argparse.ArgumentParser(
    description="""
This script donwloads parallel corpora given source and target pair language
indicators. Adapted from,
https://github.com/orhanf/blocks-examples/tree/master/machine_translation
""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-s", "--source", type=str, help="Source language",
                    default="fr")
parser.add_argument("-t", "--target", type=str, help="Target language",
                    default="en")
parser.add_argument("--source-dev", type=str, default="newstest2011.fr",
                    help="Source language dev filename")
parser.add_argument("--target-dev", type=str, default="newstest2011.en",
                    help="Target language dev filename")
parser.add_argument("--outdir", type=str, default=".",
                    help="Output directory")


def download_and_write_file(url, file_name):
    logger.info("Downloading [{}]".format(url))
    if not os.path.exists(file_name):
        path = os.path.dirname(file_name)
        if not os.path.exists(path):
            os.makedirs(path)
        u = urllib2.urlopen(url)
        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        logger.info("...saving to: %s Bytes: %s" % (file_name, file_size))
        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % \
                (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print status,
        f.close()
    else:
        logger.info("...file exists [{}]".format(file_name))


def extract_tar_file_to(file_to_extract, extract_into, names_to_look):
    extracted_filenames = []
    try:
        logger.info("Extracting file [{}] into [{}]"
                    .format(file_to_extract, extract_into))
        tar = tarfile.open(file_to_extract, 'r')
        src_trg_files = [ff for ff in tar.getnames()
                         if any([ff.find(nn) > -1 for nn in names_to_look])]
        if not len(src_trg_files):
            raise ValueError("[{}] pair does not exist in the archive!"
                             .format(src_trg_files))
        for item in tar:
            # extract only source-target pair
            if item.name in src_trg_files:
                file_path = os.path.join(extract_into, item.path)
                if not os.path.exists(file_path):
                    logger.info("...extracting [{}] into [{}]"
                                .format(item.name, file_path))
                    tar.extract(item, extract_into)
                else:
                    logger.info("...file exists [{}]".format(file_path))
                extracted_filenames.append(
                    os.path.join(extract_into, item.path))
    except Exception as e:
        logger.error("{}".format(str(e)))
    return extracted_filenames


def main():
    train_data_file = os.path.join(args.outdir, 'train_data.tgz')
    valid_data_file = os.path.join(args.outdir, 'valid_data.tgz')

    # Download europarl v7 and extract it
    download_and_write_file(TRAIN_DATA_URL, train_data_file)
    extract_tar_file_to(
        train_data_file, os.path.dirname(train_data_file),
        ["{}-{}".format(args.source, args.target)])

    # Download development set and extract it
    download_and_write_file(VALID_DATA_URL, valid_data_file)
    extract_tar_file_to(
        valid_data_file, os.path.dirname(valid_data_file),
        [args.source_dev, args.target_dev])


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('prepare_data')

    args = parser.parse_args()
    main()
