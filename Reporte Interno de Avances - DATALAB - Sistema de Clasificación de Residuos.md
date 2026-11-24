# **Reporte Interno de Avances \- DATALAB**

**Nombre del Proyecto:** Sistema de Clasificación de Residuos  
**Nombres de los Integrantes:** Emilio Juárez Ávalos, Alejandro Martínez  
**Fecha de Entrega:** 25/09/2026

## **1\. ¿Qué hicieron en este periodo?**

Definimos la arquitectura general del sistema: un módulo de basura inteligente que automatiza la separación de residuos en orgánicos, inorgánicos reciclables y no reciclables. El diseño contempla una bandeja de entrada con una microcámara en ángulo cenital que captura una imagen del residuo al momento de depositarlo. Avanzamos también en la metodología de clasificación visual, decidiendo usar Python como lenguaje principal y una Red Neuronal Convolucional (RNC) como modelo de clasificación, y comenzamos la colaboración con el equipo de Makers para el diseño de la estructura física.

## **2\. ¿Qué problemas resolvieron sobre la marcha?**

Tuvimos que definir qué arquitectura de clasificación usar: evaluamos alternativas y optamos por una RNC en lugar de una red neuronal tradicional, por su capacidad de reconocer patrones espaciales sin importar en qué parte de la bandeja caiga el residuo. También empezamos a resolver cómo coordinar la comunicación entre el software de IA y los microcontroladores que usará el equipo de Makers para mover la bandeja.

## **3\. ¿Qué aprendieron?**

Profundizamos en cómo funcionan las Redes Neuronales Convolucionales (filtros y convoluciones para el análisis espacial de imágenes) y por qué son más adecuadas que otros modelos para este tipo de clasificación visual. También aprendimos a coordinar el trabajo entre un equipo de software (Python/IA) y uno de hardware (Makers), aprovechando el ecosistema de librerías de Python para facilitar esa integración.

## **4\. Evidencia Visual**

Pendiente — se agregarán capturas del avance (código, repositorio o diseño físico) en la entrega del reporte final.