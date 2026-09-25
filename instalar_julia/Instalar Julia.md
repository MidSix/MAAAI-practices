En caso de que se haya desinstalado aquí esta la información para volver a instalarlo

--> Primero se hace lo que dice el pdf PERO, recomiendo que os saltéis solo el paso 3, en lugar de instalarlo via wizard es mejor instarlo usando el siguiente comando:
```
winget install --name Julia --id 9NJNWW8PVKMN -e -s msstore
```
este es el gestor de paquetes de windows y ya se encarga de dejar todo configurado correctamente sin problemas con juliaup ni nada. Luego, para que la autoevaluación de los tests sea mas significativa es buena idea instalar la version de Julia y de los paquetes que uso el profesor para hacer los test (experiencia adquirida de FAA xd). Para ello los siguientes comandos:

```
juliaup add 1.11.2
```
Luego
```
juliaup default 1.11.2
```
Si, la 1.11.2 es la version de Julia que se uso para realizar los tests.

--> Luego se ejecuta el archivo que instalara todas las dependencias 

