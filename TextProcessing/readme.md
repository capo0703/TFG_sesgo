# README

En este modulo se recoge la implementación del módulo de extracción de características

Para aprovechar al máximo el paralelismo, se ha implementado un sistema de colas que permite procesar los textos en paralelo.

Para ejecutar el módulo primero se debe ejecutar el administrador de colas:

```bash
python3 -m manager.py
```

luego se debe ejecutar el servicio listener:

```bash
python3 -m text_server.py
```

Para empezar el procesamiento de textos, se debe ejecutar el script `text_processing.py`:

```bash
python3 -m text_processing.py
```

La lógica de extracción de características se encuentra en el archivo `text_processing.py`, donde se definen las funciones para procesar los textos y extraer las características deseadas (en caso de que no se quiera explotar el paralelismo).
