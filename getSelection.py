import os
import numpy as np
from tqdm import tqdm

sc = ['dog_bark', 'gunshot', 'moving_motor_vehicle', 'sneeze_cough', 'footstep', 'keyboard', 'rain']

data_path = '../../drive/experiments/data/dcaseFoleySampling/'

if not os.path.exists(data_path+'selection'):
    os.makedirs(data_path+'selection')

def genSelection(data_path, sound_category, nb_sounds) : 
    import sklearn.cluster as cl    
    from sklearn.neighbors import NearestNeighbors

    name = os.path.basename(data_path)

    pck = np.load(os.path.dirname(data_path)+'/emb/'+name+'_'+sound_category+'.npy', allow_pickle=True)
    data = pck.ravel()[0]['embeddings']
    file_list = pck.ravel()[0]['file_list']

    kmeans = cl.KMeans(n_clusters = nb_sounds, random_state=0).fit(data)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(kmeans.cluster_centers_)

    u, c = np.unique(indices, return_counts=True)
    if u[c > 1].size>0:
        print('duplicate medoids detected for '+name+' '+sound_category)

    sel_file_name = os.path.dirname(data_path)+'/selection/'+os.path.basename(data_path)+'_'+sound_category+'.txt'

    file = open(sel_file_name,'w')
    for item in np.array(file_list)[indices]:
        file.write(item[0]+"\n")
    file.close()

for sci in (pbar := tqdm(range(len(sc)))):
    pbar.set_description(f"Processing {sc[sci]}")
    genSelection(data_path+'system', sc[sci], 20)
    genSelection(data_path+'dev', sc[sci], 7)