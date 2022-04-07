# In progress
import json
import torch
import numpy
import soundfile as sf

from torchaudio.vocoders import griffin_lim
from torchaudio.mel import MelTransformer
from text import TextFrontend
from model import DurIAN, BaselineDurIAN


def load_model(TTS_FRONTEND, config, checkpointPath='dur-ian-checkpoint.pt', durationModelIgnore=[], backboneModelIgnore=[]):
    model = TTS_FRONTEND(config)
    model.finetune_duration_model(checkpointPath, durationModelIgnore)
    model.finetune_backbone_model(checkpointPath, backboneModelIgnore)
    return model


def inference(outputs, config, griffin_iters=300):
    mel_fn = MelTransformer(filter_length=config['filter_length'], hop_length=config['hop_length'], win_length=config['win_length'],
                            n_mel_channels=config['n_mel_channels'], sampling_rate=config['sampling_rate'], mel_fmin=config['mel_fmin'], mel_fmax=config['mel_fmax'])
    mel_decompress = mel_fn.spectral_de_normalize(outputs)
    print("\nmel decompressshape after normalizing : ",mel_decompress.shape,mel_decompress[0][:, :-1].shape)
    
    mel_decompress = mel_decompress.transpose(1, 2)
    #print("mel decompress tensor: ",mel_decompress)
    #print("mel decompress[0] tensor: ",mel_decompress[0].cpu().numpy())
    print("\nmel decompress shape(transpose): ",mel_decompress.shape, mel_decompress[0].shape)

    #writing to PWG repo directory from where it is getting synthesized by running the code test_vocoder.sh from \
    #https://github.com/aflorithmic/ParallelWaveGAN/tree/vocoder_test
    numpy.save('/home/jovyan/ParallelWaveGAN/dump_dir/mel-spec-317000-feats.npy',mel_decompress[0][:-1, :].cpu().numpy())

    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], mel_fn.mel_basis)
    print("\nspec from mel shape ( matrix multiplication: ", spec_from_mel.shape)

    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    print("\nspec from mel shape (unsqueeze): ", spec_from_mel.shape)

    spec_from_mel = spec_from_mel * spec_from_mel_scaling
    print('\nspec_from_mel shape (scaling): ',spec_from_mel.shape, spec_from_mel[ :, :-1].shape)

    #numpy.save('/home/jovyan/ParallelWaveGAN/dump_dir/mel-spec-281000-feats.npy',spec_from_mel[0].cpu().numpy())

    griffin_iters = 300
    audio = griffin_lim(
        spec_from_mel[:, :, :-1], mel_fn.stft_fn, griffin_iters)
    return audio


def test():
    with open('configs/default.json') as f:
        config = json.load(f)

    text_frontend = TextFrontend()
    config['n_symbols'] = len(text_frontend.SYMBOLS)

    model = load_model(DurIAN, config, '/home/jovyan/Durian_aflr/LJSpeech-1.1/default_test/checkpoint_317000.pt')
    model.eval()
    inputs = [16, 5, 34, 2, 32, 5, 30, 6, 50, 25, 27, 53, 15, 26, 23, 18, 31, 43, 5, 34, 55, 55, 25, 27, 53, 18, 41, 5, 24, 5,
              34, 42, 5, 34, 15, 25, 17, 42, 44, 27, 32, 5, 44, 28, 44, 26, 16, 28, 52, 48, 34, 12, 44, 26, 15, 42, 44, 22, 44, 42, 55]
    #print('length of input: ',len(inputs))
    phonemes = text_frontend.backward(inputs)
    #print('phoneme sequence: ',phonemes)
    #print('length of phonemes: ',len(phonemes))
    inputs = torch.LongTensor([inputs])

    with torch.no_grad():
        outputs = model.inference(inputs)
    print('\nOutputs:',outputs.keys())
    #print("alignments: ",outputs['alignments'][0][60])
    postnet_outputs = outputs['postnet_outputs']
    #print("\nPostnets outputs: ",postnet_outputs)
    audio = inference(postnet_outputs, config)
    return audio

sr=22050
audio = test()[0].cpu().numpy()

sf.write('/home/jovyan/Durian_aflr/demo/demo_LJSpeech_chk_317000.wav', audio, sr)
