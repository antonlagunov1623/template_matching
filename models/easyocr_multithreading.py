import asyncio
import threading

queue = asyncio.Queue()

def threaded():
    import time
    while True:
        # time.sleep(1)
        loop.call_soon_threadsafe(queue.put_nowait, time.time())
        loop.call_soon_threadsafe(lambda: print(queue.qsize()))

@asyncio.coroutine
def async():
    while True:
        time = yield from queue.get()
        print(time)

loop = asyncio.get_event_loop()
asyncio.Task(async())
threading.Thread(target=threaded).start()
loop.run_forever()