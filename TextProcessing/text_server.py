import multiprocessing as mp
import time
import uuid
from multiprocessing.managers import BaseManager
from text_processor import TextProcessorServer

class QueueManager(BaseManager): pass
QueueManager.register('get_input_queue')
QueueManager.register('get_result_dict')

BATCH_SIZE = 64
BATCH_TIMEOUT = 0.2  # segundos

def gpu_worker(request_queue, result_dict):
    processor = TextProcessorServer()
    while True:
        batch = []
        batch_ids = []
        start = time.time()
        while len(batch) < BATCH_SIZE and (time.time() - start) < BATCH_TIMEOUT:
            try:
                req = request_queue.get(timeout=BATCH_TIMEOUT)
                batch_ids.append(req["id"])
                batch.append(req["text"])
            except Exception as e:
                break
        if batch:
            print(f"procesando batch de tamaño {len(batch)}")
            try:
                results = processor.process_batch(batch)
                for rid, res in zip(batch_ids, results):
                    result_dict[rid] = res
                print(f"batch procesado")
            except Exception as e:
                print(f"error procesando batch: {e}")
                for rid in batch_ids:
                    result_dict[rid] = {}

if __name__ == "__main__":
    manager = QueueManager(address=('localhost', 50000), authkey=b'abc123')
    manager.connect()
    input_queue = manager.get_input_queue()
    result_dict = manager.get_result_dict()
    # Lanza el worker GPU
    gpu_worker(input_queue, result_dict)
    print("worker listo")