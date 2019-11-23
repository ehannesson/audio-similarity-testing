# audioAnalysis.py
import librosa
import librosa.display
import matplotlib.pyplot as plt
import IPython.display
import numpy as np

class audioData:
    # TODO: standardize audio file names so we can auto set title/artist info
    def __init__(self, data, artist=None, track=None, sr=22050, mono=True, **kwargs):
        """
        Parameters:
            data (str): path to .wav file

            sr (int): sample rate of audio

            mono (bool): is the track mono or stereo

            **kwargs: other key word args to pass to librosa.core.load

        Attributes:
            self.data (ndarray): time-series array of audio data

            self.sr (int): sample rate of track

            self.mono (bool): track is mono if True, otherwise, stereo

            self.harmonic (ndarray): time-series of the harmonic separation
                from librosa.effects.hpss

            self.percussive (ndarray): time-series of the percussive separation
                from librosa.effects.hpss
        """
        self.data, self.sr = librosa.core.load(data, sr=sr, mono=mono, **kwargs)
        self.mono = mono

        self.track = track
        self.artist = artist

        # effects
        self.harmonic = None
        self.percussive = None

        # spectral features
        self.chroma_stft_data = None
        self.chroma_cqt_data = None
        self.chroma_cens_data = None
        self.melspectrogram_data = None
        self.mfcc_data = None
        self.spectral_centroid_data = None
        self.spectral_bandwidth_data = None
        self.spectral_contrast_data = None
        self.spectral_flatness_data = None
        self.spectral_rolloff_data = None
        self.poly_features_data = None
        self.tonnetz_data = None
        self.zero_crossing_rate_data = None

        # temporal segmentation
        self.cross_similarity = None
        self.recurrence_matrix = None
        self.recurrence_to_lag = None
        self.lag_to_recurrence = None
        self.timelag_filter = None
        self.path_enhance = None
        self.agglomerative = None
        self.subsegment = None

        ##################### Sequential Modeling ######################
        # sequence alignment
        self.dtw = None     # dynamic time warping
        self.rqa = None     # recurrence quantification analysis

        # viterbi decoding
        self.viterbi = None
        self.viterbi_discriminative = None
        self.viterbi_binary = None

        #transition matrices
        self.transition_uniform = None
        self.transition_loop = None
        self.transition_cycle = None
        self.transition_local = None

        ################################################################

        ########################### Utilities ##########################
        # matching
        self.match_intervals = None
        self.match_events = None

        # array operations
        self.softmax = None

        # miscellaneous
        self.localmax = None
        self.peak_pick = None
        self.nnls = None
        self.cyclic_gradient = None

        ################################################################

        # feature inversion (librosa.inverse.*)
        self.mel_to_stft = None
        self. mel_to_audio = None
        self.mfcc_to_mel = None
        self.mfcc_to_audio = None

        # spectrogram decomposition
        self.decomp_comp, self.decomp_activ = None, None
        self.nn_filter = None

        ####################### Important Dictionaries #########################
        # tracks the various audio objects we can have
        self.audio_type = {'data': self.data, 'harmonic': self.harmonic,
                     'percussive': self.percussive}

        # tracks the various spectral audio features we can have
        self.audio_spec = {'chroma_stft': self.chroma_stft_data,
                'chroma_cqt': self.chroma_cqt_data, 'chroma_cens': self.chroma_cens_data,
                'melspectrogram': self.melspectrogram_data, 'mfcc': self.mfcc_data,
                'spectral_centroid': self.spectral_centroid_data,
                'spectral_bandwidth': self.spectral_bandwidth_data,
                'spectral_contrast': self.spectral_contrast_data,
                'spectral_flatness': self.spectral_flatness_data,
                'spectral_rolloff': self.spectral_rolloff_data,
                'poly_features': self.poly_features_data, 'tonnetz': self.tonnetz_data,
                'zero_crossing_rate': self.zero_crossing_rate_data}

        # other stuff that doesn't quite fit above
        self.non_audio_data = {'decomp_comp': self.decomp_comp,
                               'decomp_activ': self.decomp_activ}

        # tracks the functions to create the various audio objects we can have
        self.audio_func = {'harmonic': self.hpss, 'percussive': self.hpss,
                'chroma_stft': self.chroma_stft, 'chroma_cqt': self.chroma_cqt,
                'melspectrogram': self.melspectrogram}


        self.plot_labels = {'data': 'Song Wave Form',
                            'harmonic': 'Harmonic Separation',
                            'percussive': 'Percussive Separation',
                            'chroma_stft': 'Chromagram STFT',
                            'chroma_cqt': 'Constant-Q Chromagram',
                            'melspectrogram': 'Mel-Scaled Spectrogram',
                            'chroma_cens': 'Chroma Energy Normalized',
                            'mfcc': 'Mel-Frequency Cepstral Coefficients',
                            'spectral_contrast': 'Spectral (Energy) Contrast',
                            }

    ################################ UTILITIES #################################
    def _update_dicts(self):
        """
        Refreshes these dictionaries whenever we calculate some new audio feature
        """
        # tracks the various audio objects we can have
        self.audio_type = {'data': self.data, 'harmonic': self.harmonic,
                     'percussive': self.percussive}

        # tracks the various spectral audio features we can have
        self.audio_spec = {'chroma_stft': self.chroma_stft_data,
                'chroma_cqt': self.chroma_cqt_data, 'chroma_cens': self.chroma_cens_data,
                'melspectrogram': self.melspectrogram_data, 'mfcc': self.mfcc_data,
                'spectral_centroid': self.spectral_centroid_data,
                'spectral_bandwidth': self.spectral_bandwidth_data,
                'spectral_contrast': self.spectral_contrast_data,
                'spectral_flatness': self.spectral_flatness_data,
                'spectral_rolloff': self.spectral_rolloff_data,
                'poly_features': self.poly_features_data, 'tonnetz': self.tonnetz_data,
                'zero_crossing_rate': self.zero_crossing_rate_data}

        # other stuff that doesn't quite fit above
        self.non_audio_data = {'decomp_comp': self.decomp_comp,
                               'decomp_activ': self.decomp_activ}

        return None

    def displayAudio(self, which_audio='data'):
        """
        Display's interactive audio bar in jupyter notebooks.

        Parameters:
            which (str): specifies which audio bit to play, as specified by the
                dictionary below. Default is the main audio (self.data).
        """
        try:
            return IPython.display.Audio(self.audio_type[which_audio], rate=self.sr)

        except KeyError:
            print(f'Invalid key "{which_audio}".')
            print(f'Valid options include {self.audio_type.keys()}')

        except ValueError:
            print(f'{which_audio} audio not yet computed...')
            print('Computing now...')
            # call method to compute the desired audio
            self.audio_func[which_audio]()
            print('Complete.')
            return IPython.display.Audio(self.audio_type[which_audio], rate=self.sr)

    ######################### FEATURE EXTRACTION ETC. ##########################

    def hpss(self, margin=None, returns=False):
        """
        Extracts the harmonic and percussive portions of the audio track.

        Optional Parameters:
            margin (tuple(float, float)): increasing the margin apparently
                isolates the percussive parts more -- probably just leave it
                as None. If you do want to mess with it, it seems that it is
                typically something like (1.0, 4.0) or (1.0, 8.0), or whatever
                you want it to be.

        Returns:
            None if returns=False
            self.harmonic, self.percussive if returns=True
        """
        if margin is None:
            self.harmonic, self.percussive = librosa.effects.hpss(self.data)
        else:
            self.harmonic, self.percussive = librosa.effects.hpss(self.data,
                                                                  margin=margin)
        # update dictionaries containing self.harmonic/self.percussive
        self._update_dicts()

        if returns:
            return self.harmonic, self.percussive
        else:
            return

    # SPECTRAL FEATURES
    def chroma_stft(self, S=None, norm=np.inf, n_fft=2048, hop_length=512,
                    win_length=None, window='hann', center=True, pad_mode='reflect',
                    tuning=None, n_chroma=12, returns=False):
        # docstring set below
        self.chroma_stft_data = librosa.feature.chroma_stft(self.data, sr=self.sr,
                            S=S, norm=norm, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, window=window, center=center,
                            n_chroma=n_chroma)
        # update dictionaries
        self._update_dicts()
        if returns:
            return self.chroma_stft_data
        return None

    def chroma_cqt(self, C=None, hop_length=512, fmin=None, norm=np.inf,
                threshold=0.0, tuning=None, n_chroma=12, n_octaves=7, window=None,
                bins_per_octave=None, cqt_mode='full', returns=False):
        # docstring set below
        self.chroma_cqt_data = librosa.feature.chroma_cqt(self.data, sr=self.sr,
                            C=C, hop_length=hop_length, fmin=fmin, norm=norm,
                            threshold=threshold, tuning=tuning, n_chroma=n_chroma,
                            n_octaves=n_octaves, window=window,
                            bins_per_octave=bins_per_octave, cqt_mode=cqt_mode)
        self._update_dicts()
        if returns:
            return self.chroma_cqt_data
        return None

    def melspectrogram(self, n_fft=2048, hop_length=512, win_length=None,
                        window='hann', center=True, pad_mode='reflect',
                        power=2.0, returns=False, **kwargs):
        # docstring set below
        self.melspectrogram_data = librosa.feature.melspectrogram(self.data,
                            sr=self.sr, n_fft=n_fft, hop_length=hop_length,
                            win_length=win_length, window=window, center=center,
                            pad_mode=pad_mode, power=power, **kwargs)
        if returns:
            return self.melspectrogram_data
        return None

    def decompose(self, S, n_components=None, transformer=None, sort=False,
                    fit=True, returns=False, **kwargs):
        # docstring set below
        self.decomp_comp, self.decomp_activ = librosa.decompose.decompose(S,
                            n_components=n_components, transformer=transformer,
                            sort=sort, fit=fit, **kwargs)
        if returns:
            return self.decomp_comp, self.decomp_activ
        return None


    # set docstrings
    chroma_stft.__doc__ = librosa.feature.chroma_stft.__doc__
    chroma_cqt.__doc__ = librosa.feature.chroma_cqt.__doc__
    melspectrogram.__doc__ = librosa.feature.melspectrogram.__doc__
    decompose.__doc__ = librosa.decompose.decompose.__doc__

    ################################# PLOTTING #################################

    def _plot(self, plot_func, which_audio='data', spec_plot=False, decomp=False,
                plt_title=None, db_norm=False, **kwargs):
        """
        Utility to avoid rewriting the same code over and over again.

        Parameters
        ----------
        plot_func (func): librosa plotting function

        which_audio (str): specifies which audio feature to plot

        **kwargs: keyword arguments to be passed to the plotting function
        """
        # set title
        if plt_title is None:
            plt_title = f'{self.track} -- {self.artist}\n{self.plot_labels[which_audio]}'

        if kwargs['ax'] is None:
            plt.gca().set_title(plt_title)
        else:
            kwargs['ax'].set_title(plt_title)

        if spec_plot and decomp:
            type_check = self.non_audio_data
        elif spec_plot:
            # should we be looking at spectral plots, or regular
            type_check = self.audio_spec
        else:
            type_check = self.audio_type

        try:
            # if we haven't computed specified audio feature yet
            if type_check[which_audio] is None:
                print(f'{which_audio} audio not yet computed...')
                print('Computing now...')
                try:
                    # call method to compute the desired audio feature
                    plt_data = self.audio_func[which_audio](returns=True)
                    if which_audio == 'harmonic':
                        plt_data = plt_data[0]
                    elif which_audio == 'percussive':
                        plt_data = plt_data[1]
                    # we have to update these objects with new data
                    if spec_plot:
                        # should we be looking at spectral plots, or regular
                        type_check = self.audio_spec
                    else:
                        type_check = self.audio_type

                except KeyError:
                    print(f'Invalid audio_func key {which_audio}')
                    print(f'Valid options include {audio_func.keys()}')

                print(f'Finished computing {which_audio}.')

            if db_norm:
                plt_data = librosa.amplitude_to_db(plt_data, ref=np.max)

            return plot_func(plt_data, **kwargs)

        except KeyError:
            print(f'Invalid key "{which_audio}".')
            print(f'Valid options include {type_check.keys()}')
            return None

    def waveplot(self, which_audio='data', sr=22050, x_axis='time', ax=None,
                    plt_title=None):
        # docstring set below
        return self._plot(librosa.display.waveplot, which_audio=which_audio,
                            sr=sr, x_axis=x_axis, ax=ax, plt_title=plt_title)


    def specshow(self, which_audio='melspectrogram', x_coords=None, y_coords=None,
                    x_axis=None, y_axis=None, hop_length=512, fmin=None, db_norm=True,
                    fmax=None, tuning=0.0, bins_per_octave=12, ax=None, plt_title=None):
        # docstring set below
        return self._plot(librosa.display.specshow, which_audio=which_audio,
                        spec_plot=True, x_coords=x_coords, y_coords=y_coords,
                        x_axis=x_axis, y_axis=y_axis, sr=self.sr, plt_title=plt_title,
                        hop_length=hop_length, fmin=fmin, fmax=fmax, db_norm=db_norm,
                        tuning=tuning, bins_per_octave=bins_per_octave, ax=ax)

    def decomp_plot(self, S=None, figsize=(8, 16), plt_titles=None, db_norm=True,
                    **kwargs):
        """
        S must be a spectrogram matrix. If S is not given, we first check if the
        self.decomp_comp and self.decomp_activ attributes have been assigned. If
        they have not, we calculate and plot the mel-scaled spectrogram with
        default arguments (except sort=True).

        Note that plt_titles should be a tuple, if specified, because this creates
        two subplots -- one for components, the other for activations.
        """
        if S is None:
            if self.decomp_comp is None or self.decomp_active is None:
                if self.melspectrogram_data is None:
                    self.melspectrogram()
                comp, activ = self.decompose(self.melspectrogram_data, sort=True)

        # now create plot
        fig, ax = plt.subplots(1, 2)
        # attempt to get title heading (will just be None -- None if not set)
        if title is None:
            title_header = f'{self.track} -- {self.artist}\n'
            t1 = title_header + 'Decomposition Components'
            t2 = title_header + 'Decomposition Activations'
        else:
            t1, t2 = plt_titles

        comp_plot = self._plot(librosa.display.specshow, which_audio='decomp_comp',
                                db_norm=db_norm, plt_title=t1, ax=ax[0],
                                decomp=True, spec_plot=True)
        activ_plot = self._plot(librosa.display.specshow, which_audio='decomp_activ',
                                db_norm=False, plt_title=t2, ax=ax[1],
                                decomp=True, spec_plot=True)

        return comp_plot, activ_plot

    # DOCSTRINGS
    specshow.__doc__ = librosa.display.specshow.__doc__
    waveplot.__doc__ = librosa.display.waveplot.__doc__

    ################################# ALIASES ##################################

                ################# feature aliases ################
    spec_chroma_stft = chroma_stft

                ################ plotting aliases ################
    # waveplot aliases
    waveshow = waveplot
    plot_wave = waveplot

    # specshow aliases
    specplot = specshow
    plot_spec = specshow
