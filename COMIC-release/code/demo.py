import comic
from data import load_data
from clustering_metric import clustering_evaluate
if __name__ == '__main__':

    data_name = 'bbc_sport'

    X_list, Y = load_data(data_name)
    view_size = len(X_list)
    data_size = Y.shape[0]
    print ('View Number: %d, Data Size: %d' % (view_size, data_size))
    print('V1 shape',X_list[0].shape)
    print('V2 shape',X_list[1].shape)
    gamma = 1
    pair_rate = 0.9
    
    import time
    start = time.time()

    COMIC = comic.COMIC(view_size = view_size, data_size=data_size, measure='cosine',
                        pair_rate=pair_rate, gamma=gamma, max_iter=1000)
    labels = COMIC.fit(X_list)

    elapsed = (time.time() - start)
    print ('Time used: %.2f' %elapsed)
    
    label = labels['vote']
    print ('Evaluation:')
    nmi, v_mea = clustering_evaluate(Y, label)
    print ('nmi: %.10f, v_mea: %.10f' % (nmi, v_mea))