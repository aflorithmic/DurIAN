# In progress
import json
import torch

from torchaudio.vocoders import griffin_lim
from torchaudio.mel import MelTransformer
from text import TextFrontend
from model import DurIAN, BaselineDurIAN, DurationModel
import soundfile as sf
from pydub import AudioSegment
import argparse
import numpy as np 

class InferencePipeline:
    def __init__(self, modes):
        with open('configs/default.json') as f:
            self.config = json.load(f)
        self.sr = 22050
        self.checkpoint_path = 'evelyn_test/checkpoint_45000.pt'
        self.dur_true = modes['durations']
        self.synth_true = modes['synth']
        self.text_frontend = TextFrontend()
        self.config['n_symbols'] = len(self.text_frontend.SYMBOLS)
        self.model = self.load_model(DurIAN, self.config, self.checkpoint_path)
        self.model = self.model.eval()
        self.dur_model = self.load_model(DurationModel, self.config, self.checkpoint_path)
        self.dur_model.eval()
        self.dur_out_file = 'durations.txt'
        if self.synth_true == True:
            with open(self.dur_out_file, 'w+') as o:
                pass 
        
    def load_model(self, TTS_FRONTEND, config, checkpointPath='dur-ian-checkpoint.pt', durationModelIgnore=[], backboneModelIgnore=[]):
        model = TTS_FRONTEND(self.config)
        if TTS_FRONTEND == DurIAN:
            model.finetune_duration_model(checkpointPath, durationModelIgnore)
            model.finetune_backbone_model(checkpointPath, backboneModelIgnore)
        return model

    def inference(self, outputs, config, griffin_iters=300):
        mel_fn = MelTransformer(filter_length=self.config['filter_length'], hop_length=self.config['hop_length'], win_length=self.config['win_length'],
                                n_mel_channels=self.config['n_mel_channels'], sampling_rate=self.config['sampling_rate'], mel_fmin=self.config['mel_fmin'], mel_fmax=self.config['mel_fmax'])
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

    def infer_utts(self, utt_file):
        with open(utt_file, 'r') as o:
            utt_text = o.read().splitlines()
        for line in utt_text:
            arpabet_seq = self.text_frontend.text_to_phonemes(line)
            inputs = self.text_frontend.forward(arpabet_seq)
            inputs = torch.LongTensor([inputs])
            outputs = self.model.inference(inputs)
            if self.synth_true==True:
                synthesised = self.synthesise(outputs)
                self.save_audio(synthesised)
            if self.dur_true==True:
                self.write_to_file('durations.txt', f'text: {line}\n')
                self.write_to_file('durations.txt',f'arpabet: {arpabet_seq}\n')
                self.write_to_file('durations.txt',f'durations: {outputs["durations"]}\n')
                alignment_idx =  self.get_alignments(outputs)
                self.write_to_file('durations.txt',f'alignment indices: {alignment_idx}\n')

    def synthesise(self, outputs):
        with torch.no_grad():
            postnet_outputs = outputs['postnet_outputs']
            audio = self.inference(postnet_outputs, self.config)
        return audio   

    def get_alignments(self, outputs):
        return  np.argmax(outputs["alignments"], axis=1)

    def save_audio(self, audio):
        audio = audio[0].cpu().numpy()
        sf.write('demo/demo_evelyn5.wav', audio, self.sr)

    def write_to_file(self, filename, line):
        with open(filename, 'a') as o:
            o.write(line)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Synthesis or duration prediction mode.')
    parser.add_argument('-synth', action='store_true', help='Synthesises utterance and saves audio sample to .wav file.')
    parser.add_argument('-durations', action='store_true', help='Writes duration and alignment predictions to file. Does not synthesise speech samples.')
    parser.add_argument('--utt_file', nargs='?', type=str, default='utts_to_infer.txt', const='utts_to_infer.txt', 
                        help='Path to file with utterances to infer, one utterance per line. Defaults to "./utts_to_infer.txt".')
    args = vars(parser.parse_args())
 
    UTT_FILE = args['utt_file']
    modes = {'synth': False, 'durations':False}
    if args['synth'] == True:
        modes['synth'] = True 
    if args['durations'] == True:
        modes['durations'] = True 
    infer_model = InferencePipeline(modes)
    infer_model.infer_utts(UTT_FILE)

    

    


