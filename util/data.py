import logging
import os
import zipfile

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

def extract_archive(src_dir,dst_dir,bound=114,print_progress=False):
    for i in range(bound):
        if print_progress==True:
            logger.info("{}".format(i))

            os.makedirs(dst_dir+str(i),exist_ok=True)

            with zipfile.ZipFile(src_dir + str(i) + ".zip") as r:
                r.extractall(dst_dir + str(i))
