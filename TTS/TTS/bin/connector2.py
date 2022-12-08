import argparse
import sys
from argparse import RawTextHelpFormatter

# pylint: disable=redefined-outer-name, unused-argument
from pathlib import Path

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def tts(text, OUTFILE):
    # load model manager
    out_path = Path(__file__).parent / "../tts_results/" / OUTFILE
    path = Path(__file__).parent / "../.models.json"
    manager = ModelManager(path)

    model_path = None
    config_path = None
    speakers_file_path = None
    language_ids_file_path = None
    vocoder_path = None
    vocoder_config_path = None
    encoder_path = None
    encoder_config_path = None
    use_cuda = False
    speaker_idx = None
    language_idx = None
    speaker_wav = None
    reference_wav = None
    style_wav = None
    style_text = None
    reference_speaker_name = None

    model_path, config_path, model_item = manager.download_model("tts_models/en/ljspeech/tacotron2-DCA")
    vocoder_name = model_item["default_vocoder"]
    vocoder_path, vocoder_config_path, _ = manager.download_model(vocoder_name)

    # load models
    synthesizer = Synthesizer(
        model_path,
        config_path,
        speakers_file_path,
        language_ids_file_path,
        vocoder_path,
        vocoder_config_path,
        encoder_path,
        encoder_config_path,
        use_cuda,
    )

    # kick it
    wav = synthesizer.tts(
        text,
        speaker_idx,
        language_idx,
        speaker_wav,
        reference_wav,
        style_wav,
        style_text,
        reference_speaker_name,
    )

    # save the results
    synthesizer.save_wav(wav, out_path)

if __name__ == "__main__":
    text = "Hello world"
    outfile = "test.wav"
    tts(text, outfile)
