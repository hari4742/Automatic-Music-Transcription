import multiprocessing
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
from src.processing.core.processing import AudioProcessor, MIDIProcessor
from src.processing.core.io_handler import HDF5Writer


logger = logging.getLogger(__name__)


def process_task(task, config, queue):
    """
    Process a single task (WAV + MIDI pair).

    Args:
        task (dict): Task containing file paths and metadata.
        config (DictConfig): Hydra configuration.
        queue (multiprocessing.Queue): Queue for storing results.
    """
    try:
        # Initialize processors
        audio_processor = AudioProcessor(
            sample_rate=config.processing.sample_rate,
            hop_length=config.processing.hop_length,
            cqt_bins=config.processing.cqt_bins
        )
        midi_processor = MIDIProcessor(
            resolution=config.processing.midi_resolution,
            velocity_threshold=config.processing.midi_velocity_threshold
        )

        # Process files
        audio_path = Path(config.data.raw_dir) / task["audio_filename"]
        midi_path = Path(config.data.raw_dir) / task["midi_filename"]

        cqt = audio_processor.wav_to_cqt(str(audio_path))
        pianoroll = midi_processor.midi_to_pianoroll(str(midi_path))

        # Trim to shortest dimension
        min_len = min(cqt.shape[0], pianoroll.shape[0])
        queue.put({
            "split": task["split"],
            "year": task["year"],
            "filename": audio_path.stem,
            "cqt": cqt[:min_len],
            "pianoroll": pianoroll[:min_len]
        })
    except Exception as e:
        logger.error(f"Error processing {task['audio_filename']}: {str(e)}")


def writer_process(queue, output_path):
    """
    Write processed data to the HDF5 file.

    Args:
        queue (multiprocessing.Queue): Queue for receiving results.
        output_path (str): Path to the output HDF5 file.
    """
    writer = HDF5Writer(output_path)
    while True:
        item = queue.get()
        if item is None:  # Sentinel value to stop the process
            break
        writer.write_data(
            split=item["split"],
            year=item["year"],
            filename=item["filename"],
            cqt=item["cqt"],
            pianoroll=item["pianoroll"]
        )


def run_pipeline(config):
    """
    Run the Maestro dataset processing pipeline.

    Args:
        config (DictConfig): Hydra configuration.
    """
    # Read metadata
    metadata_file = Path(config.data.metadata_file)
    df = pd.read_csv(metadata_file)
    tasks = df.to_dict("records")

    # Setup parallel processing
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    pool = multiprocessing.Pool()

    # Start writer process
    output_file = Path(config.data.processed_dir) / "maestro.hdf5"
    writer = multiprocessing.Process(
        target=writer_process,
        args=(queue, str(output_file))
    )
    writer.start()

    # Process tasks
    logger.info(
        f"Processing {len(tasks)} files with {multiprocessing.cpu_count()} workers")
    for task in tqdm(tasks, desc="Processing files", unit="file"):
        pool.apply_async(
            process_task,
            args=(task, config, queue)
        )

    # Cleanup
    pool.close()
    pool.join()
    queue.put(None)  # Signal writer to stop
    writer.join()
