## Dataset: UCDP GED 25.1


https://ucdp.uu.se/downloads/index.html#ged_global

# copiar dataset a 

datasets/GEDEvent_v25_1.csv


La idea era atribuir los eventos según su latitud y longitud a una celda de 1,8° x 1,8° de un arreglo de 100 filas por 200 columnas, agregar los 'best' de los eventos mensualmente y así se llega a una serie de 433 .csv con 20.000 valores cada uno.

Adicionalmente se aplicó un método similar a datos conseguidos de https://data.worldbank.org/ para PBI per capita y densidad de población para el año 2000, idealmente se deberían haber usado datos para todo el período del dataset de GEDEvent.

Después se planteó un patrón de relación de datos de Latitud x Longitud, en este caso usamos sólo 1 y 2 (fer fig) pero posiblemente con más cuadros en el patrón funcionaría mejor. Después la idea era hacer la reducción dimensional pero no se consiguió limpiar errores como para que funcione.


![](Untitled3.png)


https://drive.google.com/file/d/13pfb_W8FE4URuByZOnznv3mdQ6fZVUY-/view?usp=sharing

### • split ✅ ###

### • scale ✅ ###

### • stencil j+-1 k+-1 ✅ ###

### • dimensionality reduction (specific / general) ###

### • matriz de correlación con PBI y densidad de población ###

### • tabla de contingencia entre celdas cercanas ### 























Dataset de “eventos” violentos, en que se enfrentan dos actores organizados “armados”, o uno se enfrenta a civiles, con resultado de una muerte directa como mínimo.


● Eventos: 385.918

● Columnas: 49

● 1989 - 2024

● Cobertura global*

● Cada fila es un evento letal

● con fecha y geolocalización

## Variables Seleccionadas


● date_start

● date_end

● latitude, longitude

● country, region (continente)

● adm_1 (“provincia”)


● type_of_violence
1=state-based, 2=non-state,
3=one-sided

● side_a, side_b (“actores”)


● best (mejor estimación)

● deaths_civilians

## Problema a resolver: predecir mapa de calor del nivel de muertes por actores armados.

● Plantear serie de matrices de 100×200, una para cada mes del dataset.

● Dividir el mapa del mundo en una cuadrícula de 100 x 200 (1,8° x 1,8°) y agregar las estimaciones de muertes totales (best) en la serie de matrices.

● Clasificar el número de muertes en 3 categorías: gris, amarillo, naranja, rojo.

● Entrenar un modelo para clasificar cada celda del siguiente período.

● Presentar los resultados como un “mapa de calor”.

