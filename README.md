# Automatic Music Transcription

- To download dataset [check](./docs/Maestro%20Dataset%20Downloader.md)
- To start data processing pipeline [check](./docs/Maestro%20Processing%20Pipeline.md)

## TODO:

~~**Downloder:**~~

- ~~save the metadata file to `data/metadata` both when downloaded from google and hugging face~~
- ~~when downloading from hugging face, keep trace of downloaded file information save to `data/metadata` folder~~

**Audio Processing:**

- calculate sample rate automatically
- Analyse and update the logic for cqt calculation

**MIDI Processing:**

- save only 88 keys
- and that too as 0 or 1 for each key
- Analyze and update the logic for pianoroll calculation

**Configs:**

- ~~put configs into single folder~~
- ~~make a single logger~~
- separate the audio and midi processing parameters in the config file

**Pipeline:**

- change the logic of "# Trim to shortest dimension"
