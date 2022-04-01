# In progress
import json
import torch

from torchaudio.vocoders import griffin_lim
from torchaudio.mel import MelTransformer
from text import TextFrontend
from model import DurIAN, BaselineDurIAN, DurationModel
import nltk
from nltk.tokenize import WordPunctTokenizer
import soundfile as sf
from pydub import AudioSegment


def load_model(TTS_FRONTEND, config, checkpointPath='dur-ian-checkpoint.pt', durationModelIgnore=[], backboneModelIgnore=[]):
    model = TTS_FRONTEND(config)
    model.finetune_duration_model(checkpointPath, durationModelIgnore)
    model.finetune_backbone_model(checkpointPath, backboneModelIgnore)
    return model


def inference(outputs, config, griffin_iters=300):
    mel_fn = MelTransformer(filter_length=config['filter_length'], hop_length=config['hop_length'], win_length=config['win_length'],
                            n_mel_channels=config['n_mel_channels'], sampling_rate=config['sampling_rate'], mel_fmin=config['mel_fmin'], mel_fmax=config['mel_fmax'])
    mel_decompress = mel_fn.spectral_de_normalize(outputs)
    mel_decompress = mel_decompress.transpose(1, 2)
    spec_from_mel_scaling = 1000
    spec_from_mel = torch.mm(mel_decompress[0], mel_fn.mel_basis)
    spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
    spec_from_mel = spec_from_mel * spec_from_mel_scaling
    griffin_iters = 300
    audio = griffin_lim(
        spec_from_mel[:, :, :-1], mel_fn.stft_fn, griffin_iters)
    return audio


def test():
    with open('configs/default.json') as f:
        config = json.load(f)

    text_frontend = TextFrontend()
    config['n_symbols'] = len(text_frontend.SYMBOLS)
    print(config['n_symbols'])
    
    model = load_model(DurIAN, config, 'evelyn_test/checkpoint_26000.pt')
    model.eval()
    # dur_model = load_model(DurationModel, config, 'evelyn_test/checkpoint_26000.pt'
    # inputs = dur_model.inference()
    # 'we are in'
    # inputs = [16, 5, 34, 2, 32, 5, 30, 6, 50, 25, 27, 53, 15, 26, 23, 18, 31, 43, 5, 34, 55, 55, 25, 27, 53, 18, 41, 5, 24, 5,
            #   34, 42, 5, 34, 15, 25, 17, 42, 44, 27, 32, 5, 44, 28, 44, 26, 16, 28, 52, 48, 34, 12, 44, 26, 15, 42, 44, 22, 44, 42, 55]
    
    # back_to_phonemes = text_frontend.backward(inputs)
    # print(back_to_phonemes)

    nltk.download('cmudict')
    arpabet = nltk.corpus.cmudict.dict()

    with open('utts_to_infer.txt', 'r') as o:
        utt_text = o.read().splitlines()[0]
    print(utt_text)
    # token_seq =  WordPunctTokenizer().tokenize(utt_text)
    # print(token_seq)
    # arpabet_seq = ' '.join([arpabet[word.lower()] for word in token_seq])
    arpabet_seq = text_frontend.text_to_phonemes(utt_text)
    print(arpabet_seq)
    inputs = text_frontend.forward(arpabet_seq)
    inputs = torch.LongTensor([inputs])

    with torch.no_grad():
        outputs = model.inference(inputs)

    postnet_outputs = outputs['postnet_outputs']

    audio = inference(postnet_outputs, config)
    print('success')
    return audio

if __name__ == '__main__':
    audio = test()
    print(audio)

    sr = 22050
    # audio.export('evelyn_test_1.wav', format='wav')
    # 

    audio = test()[0].cpu().numpy()
    print(audio)
    #config_file = json.load('configs/default.json')
    sf.write('/home/jovyan/DurIAN/demo/demo_evelyn1.wav', audio, sr)
    # save audio as 22.050khz