import tarfile

tar = tarfile.open('method_cnn/models/model_cnn.tar.gz', "r:gz")
tar.extractall('method_cnn/models/')
tar.close()
