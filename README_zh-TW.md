# Chord-Estimation
---
[English version README](README.md)

[AI CUP 2020 - 和弦辨識競賽網站](https://aidea-web.tw/topic/43d9cc47-b70e-4751-80d3-a2d7333eb77b)

---

## 設備與環境
- 硬體：
  - CPU：i7-9700
  - GPU：RTX 2070
- 軟體：
  - Cuda：cuda_**11.0**.2_451.48_win10
  - cuDNN：cudnn-11.0-windows-x64-v**8.0.4**.30
  - TensorFlow：tensorflow-**2.4**.0

## 開發流程
1. answerAnalyze.py: 
   以 CMD 觀察數據。
2. visualization.py: 
   以 matplotlib 顯示圖形，方便觀察數據。
3. score.py: 
   瞭解評分程式的運行原理並實作。
4. main.py → processData.py: 
   開始進行模型的輸入資料預處理。
5. model.py: 
   開始建構初代模型。
6. mapping.py: 
   包含初代模型輸入資料 Y 所需要的 mapping dictionary。
   此 mapping dictionary 為 CE200 內所有歌曲之答案的集合（共 544 種可能）。
7. model.py → trainModel.py → oneFrameModel.py: 
   模組化模型，並正名為單幀輸入預測模型。
8. multiFrameModel.py: 
   建構第二代模型：多幀輸入預測模型。
9. 改良 processData.py:
   在預處理資料時切分輸入資料，讓模型得以更精確地預測答案，此舉大幅提升了正確率。

> p.s.
> oneFrameModel 與 multiFrameModel 皆只輸入「第 1 首歌的隨機 60% 資料」，來預測「第 1 首歌的剩餘 40% 資料」。
> 其中 oneFrameModel 能達到 80% 正確率，而 multiFrameModel 可達到 99.9% 正確率。

10. multiFrameModel-2.py → splitDataModel.py:
    延續多幀輸入與切分資料的概念，建構一個可以用 40% 總資料預測 60% 總資料的模型。
    Ex: CE200_sample 為 20 筆，此模型將以 8 筆為輸入資料，預測剩餘 12 筆資料。
    但因為輸入歌曲數量的增加，雖然記憶體尚可負擔多幀輸入，但無法再輸入切分資料。
    這個模型正確率只剩大約 45%，果然難題還是在於預測未知的歌曲，而非同歌曲中的部分未知資料。
11. 改良 mapping.py:
    新增並改用一個內容僅包含評分項目的 mapping dictionary。
    正確率從大約 45% 提升到了 50%，效果比想像中少許多，所以問題可能不在這裡。
12. fasterReadingModel.py:
    將資料預處理的步驟改至「餵資料進 Model 前的瞬間」，大幅降低資料讀取時間與記憶體負擔。
    
## 結果
- 得到競賽第 9 名。
- 第 1 名使用的架構在[此篇論文](https://paperswithcode.com/paper/feature-learning-for-chord-recognition-the)中。
- 第二名與我使用相同架構，但資料提取、前處理與後處理都差很多，所以成績也很差多。
- 得出心得：訓練資料的資訊含量多寡比起模型架構的優秀程度更為重要，下次遇到類似題目還是自己執行 Feature Extraction 較好。
