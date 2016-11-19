import numpy
import os

from nmt_one import train

def main(job_id, params):
    print params
    validerr = train(saveto=params['model'][0],
                                        reload_=params['reload'][0],
                                        dim_word=params['dim_word'][0],
                                        dim=params['dim'][0],
                                        n_words=params['n-words'][0],
                                        n_words_src=params['n-words'][0],
                                        decay_c=params['decay-c'][0],
                                        lrate=params['learning-rate'][0],
                                        optimizer=params['optimizer'][0],
                                        maxlen=50,
                                        batch_size=80,
                                        valid_batch_size=80,
					datasets=['/home/%s/dl4mt-tutorial/data/train.query.tok'%os.environ['USER'],
					'/home/%s/dl4mt-tutorial/data/train.reply.tok'%os.environ['USER']],
					valid_datasets=['/home/%s/dl4mt-tutorial/data/valid.query.tok'%os.environ['USER'],
					'/home/%s/dl4mt-tutorial/data/valid.reply.tok'%os.environ['USER']],
					dictionaries=['/home/%s/dl4mt-tutorial/data/index.pkl'%os.environ['USER'],
					'/home/%s/dl4mt-tutorial/data/topic.pkl'%os.environ['USER']],
                                        validFreq=5000,
                                        dispFreq=100,
                                        saveFreq=5000,
                                        sampleFreq=5000,
                                        use_dropout=params['use-dropout'][0],
                                        overwrite=False)
    return validerr

if __name__ == '__main__':
    main(0, {
        'model': ['/home/%s/dl4mt-tutorial/models/addone/model_addone.npz'%os.environ['USER']],
        'dim_word': [620],
        'dim': [1000],
        'n-words': [63000],
        'optimizer': ['adadelta'],
        'decay-c': [0.],
        'use-dropout': [False],
        'learning-rate': [0.0001],
        'reload': [False]})


