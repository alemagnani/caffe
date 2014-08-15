import random
import numpy as np
from numpy.numarray import array
from scipy.sparse import csr_matrix
import leveldb
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

import caffe.proto.caffe_pb2 as caffe_proto





def load_to_dblevel(db, X, y):
    if not isinstance(X, csr_matrix):
        raise Exception('X should be a csr matrix but its {}'.format(type(X)))
    size = X.shape[0]
    cols = int(X.shape[1])
    print 'number of cols: {}'.format(cols)

    data = X.data
    indices = X.indices
    indptr = X.indptr


    for i in xrange(size):
        if i % 500 == 0:
            print 'processed {}'.format(i)
        datum = caffe_proto.SparseDatum()
        datum.label = int(y[i])
        begin = indptr[i]
        end = indptr[i+1]
        for pos in xrange(begin,end):
            datum.float_data.append(float(data[pos]))

            datum.indices.append(int(indices[pos]))
        datum.nn = int(end - begin)
        datum.size = cols


        db.Put(str(y[i]), datum.SerializeToString())


def main():

    db_file = "/tmp/level_db_test"
    db = leveldb.LevelDB(db_file)


    categories = [
        'alt.atheism',
        'comp.graphics',
        'comp.os.ms-windows.misc',
        'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware',
        'comp.windows.x',
        'misc.forsale',
        'rec.autos',
        'rec.motorcycles',
        'rec.sport.baseball'
    ]
    # Uncomment the following to do the analysis on all the categories
    #categories = None

    print("Loading 20 newsgroups dataset for categories:")
    print(categories)

    data = fetch_20newsgroups(subset='all', categories=categories)
    print("%d documents" % len(data.filenames))
    print("%d categories" % len(data.target_names))

    ###############################################################################
    # define a pipeline combining a text feature extractor with a simple
    # classifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(max_features=1024*40,dtype=np.float32)),
        ('tfidf', TfidfTransformer())
    ])

    X = pipeline.fit_transform(data.data)
    X = X.astype(np.float32)
    y = data.target
    permutation = range(X.shape[0])
    random.seed(1)
    random.shuffle(permutation)
    print permutation[0:10]

    #permutation = permutation[0:15]

    X = X[permutation]
    #print X.data

    y = y[permutation]


    print 'about toload'
    load_to_dblevel(db,X ,y)


if __name__ == "__main__":
    main()


