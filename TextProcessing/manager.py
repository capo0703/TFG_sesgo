import multiprocessing as mp
from multiprocessing.managers import BaseManager

manager = mp.Manager()
input_queue = manager.Queue()
result_dict = manager.dict()

def get_input_queue():
    return input_queue

def get_result_dict():
    return result_dict

class QueueManager(BaseManager): pass

QueueManager.register('get_input_queue', callable=get_input_queue)
QueueManager.register('get_result_dict', callable=get_result_dict, 
                      exposed=['__getitem__', '__setitem__', '__contains__', 'get', 'keys', 'pop'])

if __name__ == "__main__":
    m = QueueManager(address=('', 50000), authkey=b'abc123')
    print("Manager remoto escuchando en puerto 50000")
    m.get_server().serve_forever()