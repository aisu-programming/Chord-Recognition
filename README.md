# Chord-Recognition
---
[中文版 README](README_zh-TW.md)

[AI CUP 2020 - Chord Recognition Competition Website](https://aidea-web.tw/topic/43d9cc47-b70e-4751-80d3-a2d7333eb77b)

Awards: https://global.turingcerts.com/en/co/cert?hash=6a1e9c33453834e3eb8f201e558c563868a308f430606bddb863fe89b6638171

---

## Environment
- Hardware：
  - CPU：i7-9700
  - GPU：RTX 2070
- Software：
  - Cuda：cuda_**11.0**.2_451.48_win10
  - cuDNN：cudnn-11.0-windows-x64-v**8.0.4**.30
  - TensorFlow：tensorflow-**2.4**.0

## Development Flow
1. `answerAnalyze.py`:

   Observe training data through CMD.
   
2. `visualization.py`:

   Observe training data much more conveniently by matplotlib.
   
3. `score.py`:
   
   Comprehend the principle of scoring method & implement the scoring program.
   
4. `main.py` → `processData.py`: 
   
   Preprocess the training data into the input data for models.
   
5. `model.py`: 
   
   Start building the fisrt version model.
   
6. `mapping.py`: 
   
   Including all required mapping dictionaries of input data Y for the first version model.
   These mapping dictionaries concludes 544 possible input data Ys in training data -- "CE200".
   
7. `model.py` → `trainModel.py` → `oneFrameModel.py`: 
   
   Modularize models & rename it to one-frame-input predicting model.
   
8. `multiFrameModel.py`: 
   
   Start building the second version model, which can use multiple frames as input data X.
   
9. Improve `processData.py`:
   
   Divide input data X during preprocessing, which significantly improved the accuracy.

> p.s.
> So far, I always input "random 40% part of one song" to predict "answers of the rest 60% part of the song".
> The oneFrameModel's accuracy is about 80%, and the multiFrameModel can achieve 99.9%.
> I then realized that this is an incorrect predicting method in the next step.

10. `multiFrameModel-2.py` → `splitDataModel.py`:

    Extends the conception of multiple-frame input & dividing input data X.
    Changing predicting pattern from "60% for a song -> answers of the rest 40%" to "60% songs -> the rest 40% songs".
    Ex: There are 20 songs in CE200_sample, so I will choose 8 songs as training data, and the rest 12 songs are for validation data. 
    After changing, the accuracy drops to 45%.
    
11. Improve `mapping.py`:
    
    Adopt new mapping dictionary which only contains scoring data Ys.
    Accuracy raise from 45% to 50%.
    
12. `fasterReadingModel.py`：
    
    Adjust the time of preprocessing to the instant before inputting data X.
    This significantly reduce the time for data reading & the GPU's RAM usage.
    
## Result
- Got the 9th place.
- The 1st place used the model in [this paper](https://paperswithcode.com/paper/feature-learning-for-chord-recognition-the).
- The second place used the same model with me, but more preprocess & Post-processing.
- Review: It seems like that information in data are much more important than the architecture of the model. Maybe I should do feature extraction on my own in the next time.
