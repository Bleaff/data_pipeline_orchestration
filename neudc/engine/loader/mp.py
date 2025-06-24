from __future__ import annotations

import multiprocessing as mp

from neudc.engine.filter import BaseFilter, ComposeFilter

from .base import BaseLoader

__all__ = ("MPLoader",)


class MPLoader(BaseLoader):
    """This class spawns multiple processes, each reading from its own data loader.
    Samples are placed on a shared Queue, and the main process forms mini-batches
    as it iterates over the object.
    """

    def __init__(
        self,
        data_loaders: list[BaseLoader],
        filters: list[BaseFilter],
        batch_size: int,
        queue_maxsize: int = 32,
    ) -> MPLoader:
        """Args:
        ----
            data_loaders (list): A list of data loader objects, each with __iter__ returning (image, metadata).
            batch_size (int): Number of samples to combine into a single batch.
            queue_maxsize (int): Max size for the multiprocessing queue.

        """
        self.filters = ComposeFilter(filters)
        self.data_loaders = data_loaders
        self.batch_size = batch_size
        self.queue = mp.Queue(maxsize=queue_maxsize)
        self.processes = []
        self.active_workers = []

    def start_workers(self) -> None:
        """Spawn one worker per data loader. Each worker iterates
        over its data loader and sends samples to the queue.
        """

        def _worker_func(
            loader_id: int,
            dataloader: BaseLoader,
            out_queue: mp.Queue,
        ) -> None:
            for metadata, image in dataloader:
                out_queue.put((loader_id, (metadata, image)))
            # Send a sentinel (None) to mark this worker is finished.
            out_queue.put((loader_id, None))

        for loader_id, loader in enumerate(self.data_loaders):
            p = mp.Process(
                target=_worker_func,
                args=(loader_id, loader, self.queue),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            self.active_workers.append(True)

    def _stop_workers(self) -> None:
        """Cleanly wait for all worker processes to finish."""
        for p, active in zip(self.processes, self.active_workers):
            if active:
                p.join()
        self.processes = []
        self.active_workers = []

    def __iter__(self):
        """A generator that yields minibatches (images, metas). Once you exhaust
        the queue (workers send sentinel), iteration ends.
        """
        if not self.processes:
            # If processes aren't running yet, start them now.
            self.start_workers()

        images, metas = [], []

        # We'll track how many workers have finished by consuming the sentinel None.
        finished_workers = 0

        while True:
            loader_id, item = self.queue.get()

            # Check for sentinel
            if item is None:
                finished_workers += 1
                self.processes[loader_id].join()
                self.active_workers[loader_id] = False
                # If all workers have sent a sentinel, stop yielding.
                if finished_workers == len(self.active_workers):
                    break
                continue

            image, meta = item
            images.append(image)
            metas.append(meta)

            if len(images) == self.batch_size:
                # Once we have enough samples, yield a batch
                yield metas, images
                images, metas = [], []

        # If there's anything left that didn't form a complete batch
        if len(images) > 0:
            yield images, metas

    def __del__(self) -> None:
        """Destructor to ensure processes are stopped if the object goes out of scope
        unexpectedly. It's safer to explicitly call self._stop_workers() in your code.
        """
        if self.processes:
            self._stop_workers()
