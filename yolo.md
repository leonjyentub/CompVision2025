---
marp: true
---

### YOLO 演進歷程與差異說明

以下是從 YOLOv1 到 YOLOv3、YOLOv5、YOLOv8 以及最新 YOLO 架構的差異與演進歷程的詳細說明，並針對每個版本的重點進行分析：

---

#### 1. YOLOv1 (2015)

- **提出者**: Joseph Redmon 等人在 2015 年提出。
- **核心概念**: 首次將目標檢測轉化為單一回歸問題（Single Shot Detection, SSD），通過一次性處理整個圖像預測邊界框和類別概率。
- **架構**:
  - 使用類似 GoogleNet 的 24 個卷積層 + 2 個全連接層。
  - 將圖像分成 7x7 網格，每個網格預測 2 個邊界框和類別概率。
- **特點**:
  - 速度快（45 FPS），適合實時應用。
  - 不使用錨框（Anchor Boxes），直接回歸邊界框坐標。
- **限制**:
  - 每個網格只能預測一個目標，難以處理重疊或小目標。
  - 定位精度較低，mAP 不如當時的兩階段方法（如 R-CNN）。
- **損失函數**: 包含坐標誤差、置信度誤差和分類誤差，使用平方和損失。

---

#### 2. YOLOv3 (2018)

- **提出者**: Joseph Redmon 和 Ali Farhadi。
- **演進**: 在 YOLOv1 和 YOLOv2（YOLO9000）的基礎上改進。
- **架構**:
  - 採用 Darknet-53 作為骨幹網路（Backbone），由 53 個卷積層組成，引入殘差結構（Residual Blocks）。
  - 使用多尺度預測（3 個不同尺度的特徵圖），每個網格預測 3 個錨框。
- **特點**:
  - 引入錨框機制，通過 K-means 聚類生成預定義錨框。
  - 支援多尺度檢測，提升小目標檢測能力。
  - 使用獨立的邏輯回歸預測物體性分數（Objectness Score）。
- **限制**:
  - 相較於兩階段方法（如 Faster R-CNN），精度仍有差距。
  - 模型複雜度增加，速度稍有下降。
- **損失函數**: 改進為二元交叉熵（Binary Cross-Entropy）用於分類和置信度預測。

---

#### 3. YOLOv5 (2020)

- **提出者**: Ultralytics 團隊（Glenn Jocher 主導），未發表正式論文。
- **演進**: 基於 YOLOv3 和 YOLOv4 的技術，轉換到 PyTorch 框架。
- **架構**:
  - 骨幹網路仍基於 CSPDarknet（Cross Stage Partial Darknet），優化計算效率。
  - 引入 Focus 層，將輸入圖像切片並重組，減少參數量。
  - Neck 使用 PANet（Path Aggregation Network）進行特徵融合。
- **特點**:
  - 首次完全用 PyTorch 實現，便於訓練和部署。
  - 提供多種模型尺寸（n/s/m/l/x），平衡速度與精度。
  - 自動學習錨框（Auto-Learning Anchor Boxes），適應不同數據集。
- **優勢**:
  - 速度快且易用，支援多種輸出格式（ONNX、CoreML 等）。
  - 社區支持強大，文檔完善。
- **限制**: 未有重大架構創新，主要改進在工程實現上。

---

#### 4. YOLOv8 (2023)

- **提出者**: Ultralytics 團隊。
- **演進**: 在 YOLOv5 基礎上進行架構和功能升級。
- **架構**:
  - 骨幹網路改進為 C2f 模塊（替代 CSP），增強特徵提取能力。
  - 採用無錨框設計（Anchor-Free），直接預測目標中心。
  - Head 部分使用解耦頭（Decoupled Head），分離物體性、分類和邊界框預測。
- **特點**:
  - 支援多任務學習：目標檢測、實例分割、姿態估計等。
  - 損失函數改進，使用 CIoU 和 DFL（Distribution Focal Loss）提升邊界框精度。
  - 提供 CLI 和 Python API，提升開發者體驗。
- **優勢**:
  - 在 COCO 數據集上精度更高（例如 YOLOv8m 達 50.2% mAP）。
  - 速度與精度平衡更佳，適應邊緣設備。
- **限制**: 未發表正式論文，部分細節需從代碼推測。

---

#### 5. 最新 YOLO 架構 (截至 2025 年 4 月)

- **代表**: YOLOv9 (2024)、YOLOv10 (2024)、YOLOv11 (2024)、YOLOv12 (2025)。
- **演進方向**:
  - **YOLOv9**: 引入 Programmable Gradient Information (PGI) 和 GELAN（Generalized Efficient Layer Aggregation Network），提升特徵提取效率和泛化能力。
  - **YOLOv10**: 由清華大學團隊開發，採用端到端設計，去除 NMS（Non-Maximum Suppression），減少延遲並降低參數量。
  - **YOLOv11**: Ultralytics 推出，支援更多任務（檢測、分割、分類、關鍵點檢測、定向邊界框），精度進一步提升（YOLOv11x 在 COCO 上達 54.7% mAP）。
  - **YOLOv12**: 引入基於注意力機制的設計，提升實時檢測精度。
- **特點**:
  - 更注重低延遲和高精度，適應邊緣計算和實時應用。
  - 結合 Transformer 和 CNN 的混合結構（例如 YOLOv11）。
  - 自動化設計（如 NAS，YOLO-NAS）優化模型結構。
- **優勢**: 在性能、效率和多功能性上全面超越早期版本。

---

### 演進歷程總結

- **從 YOLOv1 到 YOLOv3**: 從簡單回歸到多尺度檢測，引入錨框和殘差結構，逐步提升精度。
- **YOLOv5**: 轉向 PyTorch，工程化改進，易用性大幅提升。
- **YOLOv8**: 無錨框設計與多任務支持，性能和靈活性再升級。
- **最新版本**: 結合注意力機制、端到端設計和自動化搜索，追求極致效率與精度。

---

### Python 程式碼範例與 PyTorch 架構範本

以下提供從 YOLOv1 到 YOLOv8 的 PyTorch 簡化範例，並展示其演進。這些範例僅為示意，實際應用需參考官方實現。

#### YOLOv1 範例（簡化版）

```python
import torch
import torch.nn as nn

class YOLOv1(nn.Module):
    def __init__(self, num_classes=20, grid_size=7, boxes_per_cell=2):
        super(YOLOv1, self).__init__()
        self.grid_size = grid_size
        self.boxes_per_cell = boxes_per_cell
        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 簡化版省略多層結構
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * grid_size * grid_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, grid_size * grid_size * (5 * boxes_per_cell + num_classes))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        x = x.view(-1, self.grid_size, self.grid_size, 5 * self.boxes_per_cell + self.num_classes)
        return x

# 使用範例
model = YOLOv1()
input_tensor = torch.randn(1, 3, 448, 448)
output = model(input_tensor)
print(output.shape)  # [1, 7, 7, 30] (假設 20 類 + 2 框)
```

#### YOLOv3 範例（簡化版）

```python
class DarknetBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarknetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, 3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual

class YOLOv3(nn.Module):
    def __init__(self, num_classes=80, anchors=3):
        super(YOLOv3, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            DarknetBlock(32),
            # 簡化版省略多層
        )
        self.head = nn.Conv2d(32, anchors * (5 + num_classes), 1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

# 使用範例
model = YOLOv3()
input_tensor = torch.randn(1, 3, 416, 416)
output = model(input_tensor)
print(output.shape)  # 多尺度輸出需進一步處理
```

#### YOLOv5 範例（使用官方 PyTorch Hub）

```python
import torch

# 從 PyTorch Hub 加載 YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# 推理範例
img = torch.randn(1, 3, 640, 640)  # 模擬輸入圖像
results = model(img)
results.print()  # 顯示檢測結果
results.xyxy[0]  # 獲取邊界框坐標
```

#### YOLOv8 範例（使用 Ultralytics 官方庫）

```python
from ultralytics import YOLO

# 加載 YOLOv8 模型
model = YOLO('yolov8n.pt')  # 'n' 表示 nano 版本

# 推理範例
results = model.predict(source='https://ultralytics.com/images/zidane.jpg', save=True)
print(results[0].boxes.xyxy)  # 獲取邊界框坐標
print(results[0].boxes.conf)  # 置信度
print(results[0].boxes.cls)   # 類別
```

---

### 教學投影片整理（Markdown 格式）

以下是將上述內容整理成簡潔的教學投影片形式，可直接轉換為 PPT 或其他格式：

#### 投影片 1: 標題

```
YOLO 演進歷程與 PyTorch 實現
- 從 YOLOv1 到最新版本的技術進展
- 日期: 2025年4月9日
```

#### 投影片 2: YOLOv1 - 開創單階段檢測

```
- 發布: 2015
- 核心: 單一回歸問題，7x7 網格
- 架構: 24 卷積層 + 2 全連接層
- 優勢: 速度快 (45 FPS)
- 限制: 小目標、重疊目標檢測能力弱
- 程式碼片段:
  - class YOLOv1(nn.Module):
      def __init__(self): ...
```

#### 投影片 3: YOLOv3 - 多尺度與錨框

```
- 發布: 2018
- 核心: Darknet-53 + 多尺度預測
- 特點: 引入錨框，3 尺度特徵圖
- 優勢: 小目標檢測提升
- 限制: 精度仍不及兩階段方法
- 程式碼片段:
  - class DarknetBlock(nn.Module): ...
```

#### 投影片 4: YOLOv5 - PyTorch 時代

```
- 發布: 2020
- 核心: CSPDarknet + Focus 層
- 特點: PyTorch 實現，多尺寸模型
- 優勢: 易用性強，自動錨框學習
- 程式碼片段:
  - model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
```

#### 投影片 5: YOLOv8 - 無錨框與多任務

```
- 發布: 2023
- 核心: C2f 模塊 + 解耦頭
- 特點: 無錨框，多任務支持
- 優勢: 高精度 (50.2% mAP)，CLI/API
- 程式碼片段:
  - model = YOLO('yolov8n.pt')
```

#### 投影片 6: 最新版本 - YOLOv9 到 YOLOv12

```
- YOLOv9: PGI + GELAN
- YOLOv10: 端到端，無 NMS
- YOLOv11: 多任務，54.7% mAP
- YOLOv12: 注意力機制
- 趨勢: 低延遲、高精度、邊緣計算
```

#### 投影片 7: 總結與展望

```
- 演進: 從簡單回歸到多任務高效檢測
- 重點: 速度、精度、易用性逐步提升
- 未來: Transformer 融合、NAS 自動化
- 學習資源: Ultralytics GitHub
```

---

### 注意事項

- **程式碼範例**: YOLOv1 和 YOLOv3 為簡化版，實際應用需參考完整實現（如 Darknet 或 Ultralytics 倉庫）。
- **最新版本**: YOLOv12 等新模型資訊基於截至 2025 年 4 月的公開資料，可能隨後續更新有所變化。
- **投影片轉換**: 可使用 Markdown 轉 PPT 工具（如 Marp）生成正式幻燈片。

希望這些內容對您理解 YOLO 演進和實作有所幫助！若需進一步細化程式碼或投影片內容，請隨時告訴我。
