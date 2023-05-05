

import glob
import os
import numpy as np
import librosa as lr
import openl3 as l3
from tqdm import tqdm

sc = ['dog_bark', 'gunshot', 'moving_motor_vehicle', 'sneeze_cough', 'footstep', 'keyboard', 'rain']

data_path = '../../drive/experiments/data/dcaseFoleySampling/'

if not os.path.exists(data_path+'emb'):
    os.makedirs(data_path+'emb')

def genEmbedding(data_path, sound_category, compute=True) : 

    file_list = sorted(glob.glob(data_path+'/'+sc[sci]+'/*.wav'))

    # pruned_file_list = []
    # for idx, file_path in enumerate(file_list):
    #     audio, sr = lr.load(file_path)
    #     rms = lr.feature.rms(audio)
    #     crit = np.mean(rms)/(np.median(rms)+np.finfo(float).eps)
    #     if crit>1:
    #         pruned_file_list.append(file_path)
    #     else:
    #         print('removing '+file_path)

    # print(str(len(pruned_file_list))+' selected files among '+str(len(file_list))+' for class '+sound_category)

    emb_size = 512

    emb = np.zeros((len(file_list), emb_size))
    for idx, file_path in tqdm(enumerate(file_list)):
        audio, sr = lr.load(file_path)
        e, ts = l3.get_audio_embedding(audio, sr, embedding_size=emb_size, verbose=0)
        emb[idx, :] = np.mean(e)

    emb_file_name = os.path.dirname(data_path)+'/emb/'+os.path.basename(data_path)+'_'+sound_category+'.npy'

    np.save(emb_file_name, {'embeddings': emb, 'file_list': file_list})


for sci in (pbar := tqdm(range(len(sc)))):
    pbar.set_description(f"Processing {sc[sci]}")
    genEmbedding(data_path+'system', sc[sci])
    genEmbedding(data_path+'dev', sc[sci])