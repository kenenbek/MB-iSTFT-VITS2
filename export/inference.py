import sys
import os
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import librosa
import matplotlib.pyplot as plt

import os
import json
import math

import requests
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioSpeakerToneLoader, TextAudioSpeakerToneCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write
import re
from scipy import signal

# - paths
path_to_config = "/mnt/d/VITS2/config.json" # path to .json
path_to_model = "/mnt/d/VITS2/G_91000.pth" # path to G_xxxx.pth


#- text input
input = "Алар кой таштардан, арасында кум-шагыл ширелген майда жумуру таштардан, ар кандай бүртүктөгү кумдардын жана кумдуу чопонун кабатчалары менен топторунан түзүлгөн аллүбий-пролүбий чөкмө тоо тектерден турат."


# check device
if torch.cuda.is_available() is True:
    device = "cuda:0"
else:
    device = "cpu"

hps = utils.get_hparams_from_file(path_to_config)

if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
    print("Using mel posterior encoder for VITS2")
    posterior_channels = 80  # vits2
    hps.data.use_mel_posterior_encoder = True
else:
    print("Using lin posterior encoder for VITS1")
    posterior_channels = hps.data.filter_length // 2 + 1
    hps.data.use_mel_posterior_encoder = False

print(hps.data.n_speakers)
net_g = SynthesizerTrn(
    len(symbols),
    posterior_channels,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device)
_ = net_g.eval()

_ = utils.load_checkpoint(path_to_model, net_g, None)


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def langdetector(text):  # from PolyLangVITS
    try:
        lang = langdetect.detect(text).lower()
        if lang == 'ko':
            return f'[KO]{text}[KO]'
        elif lang == 'ja':
            return f'[JA]{text}[JA]'
        elif lang == 'en':
            return f'[EN]{text}[EN]'
        elif lang == 'zh-cn':
            return f'[ZH]{text}[ZH]'
        else:
            return text
    except Exception as e:
        return text

def vcss(inputstr): # single
    fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
    #fltstr = langdetector(fltstr) #- optional for cjke/cjks type cleaners
    stn_tst = get_text(fltstr, hps)

    with torch.no_grad():
        x_tst = stn_tst.to(device).unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1 / speed)[0][
                0, 0].data.cpu().float().numpy()
    write(f'./{output_dir}/output_{sid}.wav', hps.data.sampling_rate, audio)
    print(f'./{output_dir}/output_{sid}.wav Generated!')


def vcms(inputstr, sid): # multi
    fltstr = re.sub(r"[\[\]\(\)\{\}]", "", inputstr)
    #fltstr = langdetector(fltstr) #- optional for cjke/cjks type cleaners
    stn_tst = get_text(fltstr, hps)

    for idx, speaker in enumerate(speakers):
        sid = torch.LongTensor([idx]).to(device)
        with torch.no_grad():
            x_tst = stn_tst.to(device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
            audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1 / speed)[0][0,0].data.cpu().float().numpy()
        write(f'{output_dir}/{speaker}.wav', hps.data.sampling_rate, audio)
        print(f'{output_dir}/{speaker}.wav Generated!')

speed = 1
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
speakers = ["Timur", "Aiganysh"]
tones = ["neutral", "strict"]

def vcmsmt(inputstr): # multi
    stn_tst = get_text(inputstr, hps)

    for idx, speaker in enumerate(speakers):
        for idy, tone in enumerate(tones):
            sid = torch.LongTensor([idx]).to(device)
            t_id = torch.LongTensor([idy]).to(device)
            with torch.no_grad():
                x_tst = stn_tst.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
                audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, tid=t_id, noise_scale=.667, noise_scale_w=0.8, length_scale=1 / speed)[0][0,0].data.cpu().float().numpy()
            write(f'{output_dir}/{speaker}_{tone}.wav', hps.data.sampling_rate, audio)
            print(f'{output_dir}/{speaker}_{tone}.wav Generated!')


def ex_voice_conversion(sid_tgt): # dummy - TODO : further work
    #import IPython.display as ipd
    output_dir = 'ex_output'
    dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
    collate_fn = TextAudioSpeakerCollate()
    loader = DataLoader(dataset, num_workers=0, shuffle=False, batch_size=1, pin_memory=False, drop_last=True, collate_fn=collate_fn)
    data_list = list(loader)
    # print(data_list)

    with torch.no_grad():
        x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src = [x.to(device) for x in data_list[0]]
        '''
        sid_tgt1 = torch.LongTensor([1]).to(device)
        sid_tgt2 = torch.LongTensor([2]).to(device)
        sid_tgt3 = torch.LongTensor([4]).to(device)
        '''
        audio = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][0, 0].data.cpu().float().numpy()
        '''
        audio1 = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt1)[0][0, 0].data.cpu().float().numpy()
        audio2 = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt2)[0][0, 0].data.cpu().float().numpy()
        audio3 = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt3)[0][0, 0].data.cpu().float().numpy()
        '''

    write(f'./{output_dir}/output_{sid_src}-{sid_tgt}.wav', hps.data.sampling_rate, audio)
    print(f'./{output_dir}/output_{sid_src}-{sid_tgt}.wav Generated!')

    '''
    print("Original SID: %d" % sid_src.item())
    ipd.display(ipd.Audio(y[0].cpu().numpy(), rate=hps.data.sampling_rate, normalize=False))
    print("Converted SID: %d" % sid_tgt1.item())
    ipd.display(ipd.Audio(audio1, rate=hps.data.sampling_rate, normalize=False))
    print("Converted SID: %d" % sid_tgt2.item())
    ipd.display(ipd.Audio(audio2, rate=hps.data.sampling_rate, normalize=False))
    print("Converted SID: %d" % sid_tgt3.item())
    ipd.display(ipd.Audio(audio3, rate=hps.data.sampling_rate, normalize=False))
    '''

if __name__ == '__main__':
    vcmsmt(input)
