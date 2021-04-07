import math
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from queue import Queue
from typing import List

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence


class BatchGenerator(ABC):
    """Base object for generating batches. BatchGenerators should be implemented with thread safety in mind, but besides
    that, one is free to do anything as long as one implements the two methods.
    """

    @abstractmethod
    def __len__(self) -> int:
        """
        :return: Total number of batches of the to be "generated" dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        """Get the Nth batch of the generated dataset. Should be able to be done out of order and concurrently.

        :param idx: Number of batch.
        :return: Batch as an x,y tuple; x is the sample and y its label.
        """
        raise NotImplementedError

    @abstractmethod
    def __del__(self):
        """Cleanup of any objects that have to be released properly."""
        raise NotImplementedError

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def as_dataset(self) -> tf.data.Dataset:
        """Convert the BatchGenerator into a Tensorflow dataset.

        :return: Tensorflow dataset.
        """
        return tf.data.Dataset.from_generator(self.__iter__, (tf.float32, tf.float32))


class UniformBatchGenerator(BatchGenerator):
    """Batch generator that samples the dataset uniformly, loading the individual batches (of frames) directly from
    memory (after decoding) in non-sequential order. The generator returns batches as (x,y) tuples; for each sample, x
    are the first k frames of a video clip and y is the entire video clip, including x. Supports multithreading for the
    reading of the disjunctive batches.

    The dataset is structured as follows: Video data is organized on a day-to-day basis, each subset containing 24 video
    files (for each hour of the day). The sampler creates 6 disjunctive subsets of the 24 video files for each day, so
    for each batch, 4 different hours of a day are sampled. Sampling for single batch is NOT done across days due to
    the arbitrary number of thereof.

    This of course comes at a cost in performance, because the batches can be read out of order - batches in itself are
    read sequentially from the videos (that are loaded into memory after initiation), but because multiple threads
    generate batches concurrently, an overall sequential reading of all videos is unfeasible (and it would create either
    race conditions or significant overhead).
    """
    files: List[List[List[cv2.VideoCapture]]]
    file_locks: List[List[List[threading.Lock]]]

    def __init__(self, day_paths: List[str], batch_size: int = 64, sample_size: int = 5, subsample_size: int = 1):
        """Init a new UniformBatchGenerator with the given parameters. Loads all video files into memory, creates the
        necessary locks for them and structures both objects in a way that they can be accessed efficiently when
        generating new batches. Also sets up some helpful variables that are used.

        :param day_paths: List/Array of relative/absolute paths to each folder that contains 24 hours of video material.
        :param batch_size: Number of samples that are generated for a single batch.
        :param sample_size: Number of frames of which a sample consists of.
        :param subsample_size: Number of frames of which the subsample of a sample consists of (sliced off from sample).
        """
        self.files = []
        self.file_locks = []
        video_lengths = []
        for day_directory in sorted(day_paths):
            day_files = [[] for _ in range(6)]
            day_file_locks = [[] for _ in range(6)]

            if len(os.listdir(day_directory)) != 24:
                raise ValueError(day_directory + " does not contain 24 hours worth of files!")

            count = 0
            for hour_file in sorted(os.listdir(day_directory)):
                cap = cv2.VideoCapture(day_directory + hour_file)
                video_lengths.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
                day_files[count % 6].append(cap)
                day_file_locks[count % 6].append(threading.Lock())
                count += 1
            self.files.append(day_files)
            self.file_locks.append(day_file_locks)

        self.no_quarter_batches_hour = math.floor(np.amin(video_lengths) / sample_size / (batch_size / 4))
        self.no_batches_day = self.no_quarter_batches_hour * 6
        self.no_batches_total = len(self.files) * self.no_batches_day

        self.batch_size = batch_size
        self.sample_size = sample_size
        self.subsample_size = subsample_size
        self.quarter_batch_size = math.floor(batch_size / 4)
        self.quarter_batch_step_size = math.floor(batch_size / 4) * sample_size

    def __len__(self) -> int:
        """
        :return: Total number of batches of the to be "generated" dataset.
        """
        return self.no_batches_total

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        """Get the Nth batch of the generated dataset. Batches are disjunctive and do not need to be read in order.
        Decodes the video that were loaded into memory on the fly, before packaging their frames into samples (and then
        batches). Thread safe.

        :param idx: Number of batch.
        :return: Batch as an x,y tuple; x being the subsamples to the samples found in y. Normalized to [-1, 1].
        """
        day = math.floor(idx / self.no_batches_day)
        day_subset = math.floor(idx % self.no_batches_day / self.no_quarter_batches_hour)

        batch_x = []
        batch_y = []
        for file_id in range(len(self.files[day][day_subset])):
            self.file_locks[day][day_subset][file_id].acquire(blocking=True)
            file = self.files[day][day_subset][file_id]
            file.set(cv2.CAP_PROP_POS_FRAMES,
                     self.quarter_batch_step_size * (idx % self.no_batches_day % self.no_quarter_batches_hour))
            for _ in range(self.quarter_batch_size):
                sample = [file.read()[1] for _ in range(self.sample_size)]
                batch_x.append(sample[:self.subsample_size])
                batch_y.append(sample)
            self.file_locks[day][day_subset][file_id].release()

        return (np.array(batch_x).astype('float32') - 127.5) / 127.5, \
               (np.array(batch_y).astype('float32') - 127.5) / 127.5

    def __del__(self):
        """Close all opened video capture files."""
        files = np.array(self.files).flatten()
        for cap in files:
            cap.release()


class SlidingWindowBatchGenerator(BatchGenerator):
    """Batch generator that samples the dataset with a sliding window (stride 1), file by file, loading the frames of a
    video directly from memory (after decoding) in sequential order (reading of a singular batch is done in order
    however). The generator returns batches as (x,y) tuples; for each sample, x are the first k frames of a video clip
    and y is the entire video clip, including x. Supports multithreading for the reading of the disjunctive batches.

    Unlike the other BatchGenerators, this generator samples batches from explicitly passed video files (that must be
    roughly equal in length. It does not assume anything else. This generators is not meant to be used in training; this
    kind of non uniform sampling will create a bias in each batch and models might make wrong assumptions when learning
    properties of the underlying data (e.g. the first X batches are all video clips during the night, before the day
    video clips appear in the batch list). Therefore, this generator SHOULD be used for evaluation and validation alone.
    """
    files: List[cv2.VideoCapture]
    file_locks: List[threading.Lock]

    def __init__(self, file_paths: List[str], batch_size: int = 64, sample_size: int = 5, subsample_size: int = 1):
        """Init a new SlidingWindowBatchGenerator with the given parameters. Loads all video files into memory, creates
        the necessary locks for them and structures both objects in a way that they can be accessed efficiently when
        generating new batches. Also sets up some helpful variables that are used.

        :param file_paths: List of all relative/absolute paths to each equally long video file that will be used.
        :param batch_size: Number of samples that are generated for a single batch.
        :param sample_size: Number of frames of which a sample consists of.
        :param subsample_size: Number of frames of which the subsample of a sample consists of (sliced off from sample).
        """
        self.files = []
        self.file_locks = []
        video_lengths = []
        for file in sorted(file_paths):
            cap = cv2.VideoCapture(file)
            video_lengths.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            self.files.append(cap)
            self.file_locks.append(threading.Lock())

        self.no_batches_file = math.floor((np.amin(video_lengths) - (sample_size - 1)) / batch_size)
        self.no_batches_total = len(self.files) * self.no_batches_file

        self.batch_size = batch_size
        self.sample_size = sample_size
        self.subsample_size = subsample_size

    def __len__(self) -> int:
        """
        :return: Total number of batches of the to be "generated" dataset.
        """
        return self.no_batches_total

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        """Get the Nth batch of the generated dataset. Batches are disjunctive (in samples but not in frames due to
        overlap caused by the sliding window) and do not need to be read in order. Decodes the video that were loaded
        into memory at the start on the fly, before packaging their frames into samples (and then batches). Thread safe.

        :param idx: Number of batch.
        :return: Batch as an x,y tuple; x being the subsamples to the samples found in y. Normalized to [-1, 1].
        """
        file_id = math.floor(idx / self.no_batches_file)
        file = self.files[file_id]

        # cache all frames of samples in batch
        with self.file_locks[file_id]:
            file.set(cv2.CAP_PROP_POS_FRAMES, (idx % self.no_batches_file) * self.batch_size)
            frames = [file.read()[1] for _ in range(self.batch_size + self.sample_size - 1)]
            frames = (np.array(frames).astype('float32') - 127.5) / 127.5

        batch_x = []
        batch_y = []
        for i in range(self.batch_size):
            sample = frames[i:i + self.sample_size]
            batch_x.append(sample[:self.subsample_size])
            batch_y.append(sample)

        return np.array(batch_x), np.array(batch_y)

    def __del__(self):
        """Close all opened video capture files."""
        files = np.array(self.files).flatten()
        for cap in files:
            cap.release()


class BatchLoader:
    """Wrapper around :py:class:`batch_generator.BatchGenerator`, to allow concurrent loading of batches from the
    generator into a shared cache in the background, while the main thread is able to request batches from that cache.

    This loader simulates the functionalities of some of the Keras Sequence API, that is used for batch loading, but
    unlike the Sequence API it can be used in custom training loops that do not utilize the fit() method and thus can't
    use the former API.
    """
    batch_generator: BatchGenerator
    batch_q: Queue
    no_remaining_batches: int

    workers: List[threading.Thread]
    worker_ret_vals: Queue

    def __init__(self, batch_generator: BatchGenerator, max_queue_size=16, no_workers: int = 4):
        """Init a new BatchLoader with the given parameters. Spawns requested number of worker threads, computes the
        necessary list of batch IDs that each worker has to process, and inits batch queue for the workers to load
        batches to.

        :param batch_generator: Generator to load batches from.
        :param max_queue_size: Size of cache for loaded batches.
        :param no_workers: Number of threads to load batches from generator.
        """
        self.batch_generator = batch_generator
        self.batch_q = Queue(max_queue_size)
        self.no_remaining_batches = len(batch_generator)

        self.workers = []
        self.worker_ret_vals = Queue()
        for thread_id in range(no_workers):
            batch_ids = [batch_id for batch_id in range(thread_id, len(batch_generator), no_workers)]
            t = threading.Thread(target=self.worker, args=(batch_ids,))
            t.start()
            self.workers.append(t)

    def __len__(self):
        """
        :return: Total number of batches that can be loaded from the batch generator.
        """
        return len(self.batch_generator)

    def __del__(self):
        """Shut down all workers and empty queue."""
        self.shutdown_workers()

    def worker(self, batch_ids: List[int]) -> None:
        """Method for the worker threads. Worker threads shuffle their batch id subsets before requesting
        the batches respective to those IDs to reduce collisions across worker threads. Batches are concurrently loaded
        into the batch queue while the main thread is able to get batches from it. Blocking in case the batch cache has
        reached its full capacity.

        :param batch_ids: List of all batches (IDs) that a worker thread will load.
        :return: None, but return values are stored in a queue (worker_ret_vals). Contains the total number of samples
        loaded, the number of batches loaded, and the number of batches that should have been loaded.
        """
        random.shuffle(batch_ids)
        no_batches_loaded = 0
        no_samples_loaded = 0

        for idx in batch_ids:
            batch = self.batch_generator[idx]
            self.batch_q.put(batch)
            no_batches_loaded += 1
            no_samples_loaded += len(batch[0])

        self.worker_ret_vals.put((no_samples_loaded, no_batches_loaded, len(batch_ids)))

    def get_batch(self) -> (np.ndarray, np.ndarray):
        """Gets a loaded batch. Blocks in case the batch cache is empty until a worker has loaded one batch into it.
        Not thread safe.

        :raises StopIteration: If no more batches can be loaded from generator and cache is empty.
        :return: Batch as an x,y tuple; x is the sample and y its label.
        """
        if self.no_remaining_batches == 0:
            raise StopIteration("All batches already loaded!")
        self.no_remaining_batches -= 1
        return self.batch_q.get()

    def shutdown_workers(self, summary=False) -> None:
        """Optional. Used to cleanly shutdown the worker threads by fast forwarding through their batches.

        :param summary: Flag whether to print out a summary.
        """
        no_rest = 0
        for _ in range(self.no_remaining_batches):
            self.get_batch()
            no_rest += 1

        for t in self.workers:
            t.join()

        if summary:
            no_samples_loaded_t = [i for i, _, _ in list(self.worker_ret_vals.queue)]
            no_batches_loaded_t = [i for _, i, _ in list(self.worker_ret_vals.queue)]
            no_batches_assigned_t = [i for _, _, i in list(self.worker_ret_vals.queue)]

            print("------------------------- BatchLoader Summary START -------------------------",
                  "Total number of batches: " + str(len(self.batch_generator)),
                  "Number of not requested batches at shutdown: " + str(no_rest),
                  "",
                  "Number of worker threads: " + str(len(self.workers)),
                  "Number of batches assigned to each thread: " + str(no_batches_assigned_t),
                  "Number of batches loaded by each thread: " + str(no_batches_loaded_t),
                  "Number of samples loaded by each thread: " + str(no_samples_loaded_t),
                  "-------------------------- BatchLoader Summary END --------------------------",
                  sep='\n')


class BatchSequence(Sequence):
    """Wrapper around :py:class:`batch_generator.BatchGenerator`, that supports the Keras Generator & Sequence API to
    load batches concurrently and with thread safety in mind.
    """
    batch_generator: BatchGenerator

    def __init__(self, batch_generator: BatchGenerator):
        """
        :param batch_generator: Generator to load batches from. Generator must be stateless and thread safe.
        """
        self.batch_generator = batch_generator

    def __len__(self) -> int:
        """
        :return: Total number of batches that can be loaded from the batch generator.
        """
        return len(self.batch_generator)

    def __getitem__(self, idx: int) -> (np.ndarray, np.ndarray):
        """
        :param idx: Number of batch.
        :return: Batch as an x,y tuple; x is the sample and y its label.
        """
        return self.batch_generator[idx]

    def as_dataset(self) -> tf.data.Dataset:
        """
        :return: Tensorflow dataset.
        """
        return tf.data.Dataset.from_generator(self.__iter__, (tf.float32, tf.float32))


def test_batch_loader(no_threads: int = 4) -> None:
    """Test function for the BatchLoader; a single BatchGenerator is created for a BatchLoader. Will request batches
    from loader until that one is exhausted. Afterwards, everything is shut down in a clean manner..

    :param no_threads: Number of worker threads of BatchLoader.
    """
    base_path = "../../data/upCam/preprocessed_256_128/"
    days = ["00", "01", "02", "03", "04"]
    day_paths = [base_path + day + "/" for day in days]

    uniform_sampler = UniformBatchGenerator(day_paths=day_paths, batch_size=64, sample_size=10, subsample_size=5)
    batch_loader = BatchLoader(uniform_sampler, 128, no_threads)
    for _ in range(len(batch_loader)):
        batch_loader.get_batch()
    batch_loader.shutdown_workers(summary=True)


if __name__ == '__main__':
    for k in 1, 2, 4, 8:
        start_time = time.time()
        print("-------------------- Running batch loader with " + str(k) + " Threads...")
        test_batch_loader(k)
        print("---------- Run completed after %s seconds." % (time.time() - start_time))
