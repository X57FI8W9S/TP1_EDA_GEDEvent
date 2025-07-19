## Dataset: UCDP GED 25.1


https://ucdp.uu.se/downloads/index.html#ged_global

# copiar dataset a 

datasets/GEDEvent_v25_1.csv


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


la idea sería que quede así:
![](https://github.com/X57FI8W9S/TP1_EDA_GEDEvent/blob/main/notebooks/img/Screenshot.png)


El análisis exploratorio del conjunto de datos UCDP GEDEvent 25.1 permitió obtener una visión general de la violencia organizada con resultado letal en distintas regiones del mundo. Se observa que la región de Medio Oriente concentra la mayor cantidad de eventos, más de 122.000, seguida por Asia y África. Los países más afectados son Siria, Afganistán y Ucrania. El Gobierno de Siria aparece como el actor A más frecuente, mientras que los insurgentes sirios y la población civil son los actores B más frecuentes. El tipo de violencia más común es el conflicto estatal (70%), seguido por la violencia unilateral (15%) y los conflictos no estatales (14%).


