import torch
import numpy as np



class FourierTransform:
    def __init__(self,
                 fft_bins=2048,
                 win_length_ms=40,
                 frame_rate_hz=100,
                 causal=False,
                 preemphasis=0.0,
                 sample_rate=44100,
                 normalized=False):
        self.sample_rate = sample_rate
        self.frame_rate_hz = frame_rate_hz
        self.preemphasis = preemphasis
        self.fft_bins = fft_bins
        self.win_length = int(sample_rate * win_length_ms / 1000)
        self.hop_length = int(sample_rate / frame_rate_hz)
        self.causal = causal
        self.normalized = normalized
        if self.win_length > self.fft_bins:
            print('FourierTransform Warning: fft_bins should be larger than win_length')

    def _convert_format(self, data, expected_dims):
        if not type(data) == torch.Tensor:
            data = torch.Tensor(data)
        if len(data.shape) < expected_dims:
            data = data.unsqueeze(0)
        if not len(data.shape) == expected_dims:
            raise Exception(f"FourierTransform: data needs to be a Tensor with {expected_dims} dimensions but got shape {data.shape}")
        return data

    def _preemphasis(self, audio):
        if self.preemphasis > 0:
            return torch.cat((audio[:, 0:1], audio[:, 1:] - self.preemphasis * audio[:, :-1]), dim=1)
        return audio

    def _revert_preemphasis(self, audio):
        if self.preemphasis > 0:
            for i in range(1, audio.shape[1]):
                audio[:, i] = audio[:, i] + self.preemphasis * audio[:, i-1]
        return audio

    def _magphase(self, complex_stft):
        mag, phase = torch.functional.magphase(complex_stft, 1.0)
        return mag, phase

    def stft(self, audio):
        '''
        wrapper around th.stft
        audio: wave signal as th.Tensor
        '''
        hann = torch.hann_window(self.win_length)
        hann = hann.cuda() if audio.is_cuda else hann
        spec = torch.stft(audio, n_fft=self.fft_bins, hop_length=self.hop_length, win_length=self.win_length,
                       window=hann, center=not self.causal, normalized=self.normalized)
        return spec.contiguous()

    def complex_spectrogram(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: th.Tensor of size channels x frequencies x time_steps (channels x y_axis x x_axis)
        '''
        self._convert_format(audio, expected_dims=2)
        audio = self._preemphasis(audio)
        return self.stft(audio)

    def magnitude_phase(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: tuple containing two th.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        stft = self.complex_spectrogram(audio)
        return self._magphase(stft)

    def mag_spectrogram(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: magnitude spectrum as th.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        return self.magnitude_phase(audio)[0]

    def power_spectrogram(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: power spectrum as th.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        return torch.pow(self.mag_spectrogram(audio), 2.0)

    def phase_spectrogram(self, audio):
        '''
        audio: wave signal as th.Tensor
        return: phase spectrum as th.Tensor of size channels x frequencies x time_steps for magnitude and phase spectrum
        '''
        return self.magnitude_phase(audio)[1]

    def mel_spectrogram(self, audio, n_mels):
        '''
        audio: wave signal as th.Tensor
        n_mels: number of bins used for mel scale warping
        return: mel spectrogram as th.Tensor of size channels x n_mels x time_steps for magnitude and phase spectrum
        '''
        spec = self.power_spectrogram(audio)
        mel_warping = torchaudio.transforms.MelScale(n_mels, self.sample_rate)
        return mel_warping(spec)

    def complex_spec2wav(self, complex_spec, length):
        '''
        inverse stft
        complex_spec: complex spectrum as th.Tensor of size channels x frequencies x time_steps x 2 (real part/imaginary part)
        length: length of the audio to be reconstructed (in frames)
        '''
        complex_spec = self._convert_format(complex_spec, expected_dims=4)
        hann = torch.hann_window(self.win_length)
        hann = hann.cuda() if complex_spec.is_cuda else hann
        wav = torchaudio.functional.istft(complex_spec, n_fft=self.fft_bins, hop_length=self.hop_length, win_length=self.win_length, window=hann, length=length, center=not self.causal)
        wav = self._revert_preemphasis(wav)
        return wav

    def magphase2wav(self, mag_spec, phase_spec, length):
        '''
        reconstruction of wav signal from magnitude and phase spectrum
        mag_spec: magnitude spectrum as th.Tensor of size channels x frequencies x time_steps
        phase_spec: phase spectrum as th.Tensor of size channels x frequencies x time_steps
        length: length of the audio to be reconstructed (in frames)
        '''
        mag_spec = self._convert_format(mag_spec, expected_dims=3)
        phase_spec = self._convert_format(phase_spec, expected_dims=3)
        complex_spec = torch.stack([mag_spec * torch.cos(phase_spec), mag_spec * torch.sin(phase_spec)], dim=-1)
        return self.complex_spec2wav(complex_spec, length)

    def loss(self, voice_signals, gt_voice_signals, debug=False):
        ## Magnitude
        l2 = torch.mean((voice_signals - gt_voice_signals).pow(2))

        ## Phase
        _transform = lambda x: self.stft(x.reshape(-1, x.shape[-1]))
        data, target = _transform(voice_signals).view(-1, 2), _transform(gt_voice_signals).view(-1, 2)
        target_energy = torch.sum(torch.abs(target), dim=-1)
        pred_energy = torch.sum(torch.abs(data.detach()), dim=-1)
        
        target_mask = target_energy > 0.1 * torch.mean(target_energy)
        pred_mask = pred_energy > 0.1 * torch.mean(target_energy)
        
        indices = torch.nonzero(target_mask * pred_mask).view(-1)
        data, target = torch.index_select(data, 0, indices), torch.index_select(target, 0, indices)
        data_angles, target_angles = torch.atan2(data[:, 0], data[:, 1]), torch.atan2(target[:, 0], target[:, 1])
        phase_loss = torch.abs(data_angles - target_angles)
        phase_loss = np.pi - torch.abs(phase_loss - np.pi)
        phase_loss = torch.mean(phase_loss)

        loss = l2 + 0.1 * phase_loss
        return loss
