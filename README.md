Modelo principal: LightGBM

Evaluacion de modelos.
Clasification report:
F1 score: Sirve para evaluar combinando solo valor de 0 a 1 la presicion y la exhaustividad mediante su media armonica. da un equilibrio entre falsos positivos y falsos negativos, se enfoca en desaserse de clases desequilibradas.
Roc_Auc_Score: sirve para medir que tan bien distingue un modelo de clasificacion binaria entre dos clases a traves de todos los umbrales de desicion posibles.
Train_Test_Split: divide un conjunto de datos en dos partes un bloque para entrenar el algoritmo y otro bloque reservado para ponerlo a prueba.

El algoritmo entrena cientos de arboles poco profundos uno tras otro. El primer arbol hace una estimacion inicial si el usuario le gustara el anime, el segundo arbol analiza en que filas se equivoco el primero y aprende a reglas para cometer menos errores hasta minimizar la perdida algoritmica. Para procesar millones de filas sin cosumir demaciada memoria agrupa las variables en intervalos discretos y expande los arboles dividiendo la hoja que logra menos errores y aprende sus patrones.Ordena las categorias segun su proporcion de calificaciones altas en cada divicion del arbol y encuentra el mejor punto. Pasa toda la informacion de los arboles suma los puntajes y los transforma mediante una funcion sigmoide para devolver una probabilidad de 0 y 1 que el usuario otorgue una calificacion mayor o igual a 8.

Accuracy: Sirve para medir la proporcion de prediccioes correctas que realiza un modelo de clasificacion sobre el total de casos evaluados.

Resultados del proyecto