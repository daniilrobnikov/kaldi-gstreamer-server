import unittest
import wave
from pathlib import Path

import torch
import torchaudio

import sherpa


def decode(
    recognizer: sherpa.OnlineRecognizer,
    s: sherpa.OnlineStream,
    samples: torch.Tensor,
):
    expected_sample_rate = 16000

    tail_padding = torch.zeros(int(16000 * 0.3), dtype=torch.float32)

    chunk = int(0.2 * expected_sample_rate)  # 0.2 seconds

    start = 0
    last_result = ""
    while start < samples.numel():
        end = start + chunk
        s.accept_waveform(expected_sample_rate, samples[start:end])
        start = end

        while recognizer.is_ready(s):
            recognizer.decode_stream(s)
            result = recognizer.get_result(s).text
            if last_result != result:
                last_result = result
                print(result)

    s.accept_waveform(expected_sample_rate, tail_padding)
    s.input_finished()

    while recognizer.is_ready(s):
        recognizer.decode_stream(s)
        result = recognizer.get_result(s).text
        if last_result != result:
            last_result = result
            print(result)


d = "/tmp/icefall-models"
# Please refer to
# https://k2-fsa.github.io/sherpa/cpp/pretrained_models/online_transducer.html
# to download pre-trained models for testing


class TestOnlineRecognizer(unittest.TestCase):
    def test_icefall_asr_librispeech_pruned_transducer_stateless7_streaming_2022_12_29(
        self,
    ):
        nn_model = f"{d}/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/exp/cpu_jit.pt"
        tokens = f"{d}/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/tokens.txt"
        lg = f"{d}/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/data/lang_bpe_500/LG.pt"
        wave = f"{d}/icefall-asr-librispeech-pruned-transducer-stateless7-streaming-2022-12-29/test_wavs/1089-134686-0001.wav"

        if not Path(encoder_model).is_file():
            print(f"{nn_model} does not exist")
            print(
                "skipping test_icefall_asr_librispeech_pruned_transducer_stateless7_streaming_2022_12_29()"
            )
            return

        feat_config = sherpa.FeatureConfig()
        expected_sample_rate = 16000

        samples, sample_rate = torchaudio.load(wave)
        assert sample_rate == expected_sample_rate, (
            sample_rate,
            expected_sample_rate,
        )
        samples = samples.squeeze(0)

        feat_config.fbank_opts.frame_opts.samp_freq = expected_sample_rate
        feat_config.fbank_opts.mel_opts.num_bins = 80
        feat_config.fbank_opts.frame_opts.dither = 0

        print("--------------------modified beam search--------------------")
        config = sherpa.OnlineRecognizerConfig(
            nn_model=nn_model,
            tokens=tokens,
            use_gpu=False,
            feat_config=feat_config,
            decoding_method="modified_beam_search",
            num_active_paths=4,
            chunk_size=32,
        )

        recognizer = sherpa.OnlineRecognizer(config)

        s = recognizer.create_stream()

        decode(recognizer=recognizer, s=s, samples=samples)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)
torch._C._set_graph_executor_optimize(False)
if __name__ == "__main__":
    unittest.main()
