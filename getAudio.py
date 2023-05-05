import shutil
import os
import random
import copy
import numpy as np
import soundfile as sf
import librosa as lr
from tqdm.notebook import tqdm


sc = ['dog_bark', 'gunshot', 'moving_motor_vehicle', 'sneeze_cough', 'footstep', 'keyboard', 'rain']
selector = {'dog_bark': [2, 5, 6], 
            'gunshot': [2, 6], 
            'moving_motor_vehicle': [0, 1, 3, 4], 
            'sneeze_cough': [2, 6], 
            'footstep': [0, 2, 6], 
            'keyboard': [0, 2, 3, 6], 
            'rain': [0, 1, 3, 4]
            }

data_path = '../../drive/experiments/data/dcaseFoleySampling/'

for sci in (pbar := tqdm(range(len(sc)))):
    pbar.set_description(f"Processing {sc[sci]}")

    with open(data_path+'selection/system_'+sc[sci]+'.txt') as f:
        file_names = [line.rstrip('\n') for line in f]
    for k, f in enumerate(file_names):
        shutil.copyfile(f, data_path+'stimuli/'+sc[sci]+'/'+sc[sci]+'_system_'+str(k)+'.wav')

    with open(data_path+'selection/dev_'+sc[sci]+'.txt') as f:
        file_names = [line.rstrip('\n') for line in f]
    for k, f in enumerate(file_names):
        shutil.copyfile(f, data_path+'stimuli/'+sc[sci]+'/'+sc[sci]+'_reference_'+str(k)+'.wav')

    for k in range(3):
        shutil.copyfile(file_names[k], data_path+'stimuli/'+sc[sci]+'/'+sc[sci]+'_gqgf'+str(k)+'.wav')

    scr = copy.deepcopy(sc)
    scr = [scr[i] for i in selector[sc[sci]]]
    for k in range(3):
        scni = random.choice(list(range(len(scr))))

        with open(data_path+'selection/dev_'+scr[scni]+'.txt') as f:
            other_file_names = [line.rstrip('\n') for line in f]

        shutil.copyfile(random.choice(other_file_names), data_path+'stimuli/'+sc[sci]+'/'+sc[sci]+'_gqbf'+str(k)+'.wav')

    for k in range(3):
        shutil.copyfile(file_names[3+k], data_path+'stimuli/'+sc[sci]+'/'+sc[sci]+'_gqgf'+str(k)+'.wav')

    audio, sr = lr.load(file_names[3])
    factor = 6
    audio = lr.resample(audio, sr, sr//factor)
    audio = lr.resample(audio, sr, sr*factor)
    sf.write(data_path+'stimuli/'+sc[sci]+'/'+sc[sci]+'_bqgf_0.wav', audio, sr)

    audio, sr = lr.load(file_names[4])
    factor = 20
    audio = audio+np.random.randn(audio.shape[0])*np.max(np.abs(audio))/factor
    sf.write(data_path+'stimuli/'+sc[sci]+'/'+sc[sci]+'_bqgf_1.wav', audio, sr)

    audio, sr = lr.load(file_names[5])
    factor = 20
    audio = np.clip(audio, np.min(audio)/factor, np.max(audio)/factor)
    sf.write(data_path+'stimuli/'+sc[sci]+'/'+sc[sci]+'_bqgf_2.wav', audio, sr)