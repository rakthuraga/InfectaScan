# InfectaScan

Wastewater epidemiology has existed for decades. However, it lacks specificity and speed. We want to improve on it by using Deep Learning and optical imaging to analyze wastewater from individual patients, giving healthcare providers insights into their patients' microbiome health and early detection of opportunistic hospital-acquired infections. Signs in the stool may result within 3 hours of colonization, a massive improvement over the days or weeks in which other symptoms would become apparent. This technology could lower the negative impact of prescribing the wrong antibiotics. With 44% of safety errors specifically attributed to medication errors like with antibiotics, especially in the elderly and in racial minorities, InfectaScan could massively improve patient safety, quality of life, and equity.

## Reproducing Results

The "training.ipynb" contains the code used to set up and train the AI model. The pre-trained AI model can be run in a container. Docker commands:
```
docker build -t infectascan .
docker run -it -p 8080:8080 infectascan
```
The "demo.ipynb" Python notebook contains test data and visualization tools. The cells for testing Google Cloud require a deployed Vertex AI endpoint and authorization information.

## Technologies

InfectaScan uses a Convolutional Neural Network trained on the DIBaS dataset. InfectaScan supports local instances with a React-based interface as well as remote Docker instances, meaning it can scale from readily available consumer computers to the entirety of a healthcare institution's computing resources. Cloud computing with Google Cloud has also been implemented.

## References

```
@article{zielinski2017,
	title={Deep learning approach to bacterial colony classification},
	author={Zieli{\'n}ski, Bartosz and Plichta, Anna and Misztal, Krzysztof and Spurek, Przemys{\l}aw and Brzychczy-W{\l}och, Monika and Ocho{\'n}ska, Dorota},
	journal={PloS One},
	volume={12},
	number={9},
	pages={e0184554},
	year={2017},
	publisher={Public Library of Science San Francisco, CA USA}
}
```
