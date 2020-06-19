import logging
import zipfile
import os

ARCHIVE_DIR = "./WikipediaImages/Archives/"
IMAGE_DIR = "./WikipediaImages/Images/"

logger = logging.getLogger("awi")
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    logger.info("ZIPファイルの解凍を開始します。")
    for i in range(0, 114):
        logging.info("{}".format(i))

        os.mkdir(IMAGE_DIR + str(i))

        with zipfile.ZipFile(ARCHIVE_DIR + str(i) + ".zip") as r:
            r.extractall(IMAGE_DIR + str(i))

    logger.info("ZIPファイルの解凍が終了しました。")
