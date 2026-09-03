"""In-memory queue: the webhook answers 200 instantly, work happens behind it.

Meta redelivers an event when the endpoint is slow, which is exactly how bots
end up answering twice — so no LLM call ever runs inside the request. Each
number gets its own lock, so one person's messages are answered in order.
"""

from __future__ import annotations

import asyncio
import logging

from app.services.handler import MessageHandler
from app.whatsapp.schemas import IncomingMessage

logger = logging.getLogger(__name__)


class Dispatcher:
    def __init__(self, handler: MessageHandler, workers: int = 4, max_queue: int = 1000) -> None:
        self.handler = handler
        self.workers = max(1, workers)
        self._queue: asyncio.Queue[IncomingMessage] = asyncio.Queue(maxsize=max_queue)
        self._tasks: list[asyncio.Task] = []
        self._locks: dict[str, asyncio.Lock] = {}
        self._running = False

    # ----------------------------------------------------------- lifecycle
    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks = [
            asyncio.create_task(self._worker(i), name=f"wabot-worker-{i}")
            for i in range(self.workers)
        ]
        logger.info("Dispatcher started with %d workers", self.workers)

    async def stop(self, drain_timeout: float = 10.0) -> None:
        """Let in-flight messages finish, then cancel the workers."""
        if not self._running:
            return
        self._running = False
        try:
            await asyncio.wait_for(self._queue.join(), timeout=drain_timeout)
        except asyncio.TimeoutError:
            logger.warning("Queue did not drain in %.1fs — shutting down anyway", drain_timeout)

        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("Dispatcher stopped")

    # --------------------------------------------------------------- queue
    async def submit(self, message: IncomingMessage) -> bool:
        """Enqueue without blocking. False means the queue is full."""
        try:
            self._queue.put_nowait(message)
            return True
        except asyncio.QueueFull:
            logger.error("Queue is full — dropping message %s", message.message_id)
            return False

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def _lock_for(self, wa_id: str) -> asyncio.Lock:
        """One lock per number keeps a single thread strictly sequential."""
        lock = self._locks.get(wa_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[wa_id] = lock
        return lock

    async def _worker(self, index: int) -> None:
        while True:
            message = await self._queue.get()
            try:
                async with self._lock_for(message.from_number):
                    await self.handler.handle(message)
            except Exception:
                # One bad message must never kill a worker.
                logger.exception("Worker %d crashed handling %s", index, message.message_id)
            finally:
                # CancelledError is a BaseException, so it skips the except above
                # and still marks the task done here before propagating.
                self._queue.task_done()
