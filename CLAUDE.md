# Expression Detect

人脸表情检测项目：输入人脸视频 → 检测 52 个 ARKit blendshapes + 468 个 3D 面部关键点 → 输出标注视频 + 数据文件供 Blender 导入。

## 常用命令

```bash
uv sync                                          # 安装依赖
uv run python -m src.main <video.mp4>             # 运行（带实时预览）
uv run python -m src.main <video.mp4> --no-preview # 无预览（headless）
uv run python -m src.main <video.mp4> --show-depth-map  # 带深度图侧边栏
uv run python -m src.main <video.mp4> --export-format csv  # CSV 导出
```

## 项目结构

```
src/
├── main.py           # CLI 入口，argparse 参数解析在 config.py
├── config.py         # PipelineConfig dataclass + parse_args()
├── pipeline.py       # 主流程：视频读取 → 检测 → 可视化 → 导出
├── detector.py       # MediaPipe FaceLandmarker 封装（VIDEO 模式）
├── visualizer.py     # 绘制面部网格、blendshape 文本叠加、深度图
├── exporter.py       # JSON/CSV 导出，52 个 ARKit blendshape 名称定义在此
├── model_manager.py  # 模型自动下载（首次运行）
└── types.py          # FrameResult, VideoMetadata 数据结构
scripts/
└── blender_import.py # Blender 端导入脚本（在 Blender 脚本编辑器中运行）
models/               # 模型文件 face_landmarker_v2_with_blendshapes.task (~3.8MB)
output/               # 默认输出目录（_annotated.mp4 + _blendshapes.json）
```

## 技术栈

- **Python 3.12+**, 包管理用 **uv**
- **MediaPipe 0.10.32** — FaceLandmarker Task API
- **OpenCV** — 视频读写 + 可视化绘制
- **NumPy** — 数组操作

## MediaPipe 0.10.32 API 注意事项

`mp.solutions` 模块已移除，必须用新 API：

```python
from mediapipe.tasks.python.vision import drawing_utils, drawing_styles
from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
```

- `draw_landmarks()` 接受 `list[NormalizedLandmark]`（不是 protobuf）
- 连接集：`FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION` / `FACE_LANDMARKS_CONTOURS`
- 虹膜：`FACE_LANDMARKS_LEFT_IRIS + FACE_LANDMARKS_RIGHT_IRIS`（没有合并的 IRISES 常量）

## 数据流

1. `pipeline.py` 用 `cv2.VideoCapture` 逐帧读取视频
2. `detector.py` 每帧做 BGR→RGB 转换，调用 `FaceLandmarker.detect_for_video()`（timestamp 必须单调递增）
3. 返回 `FrameResult`：landmarks (468×3), blendshapes (52 个 name→score), transformation_matrix (4×4)
4. `visualizer.py` 在帧上绘制网格 + 文字叠加
5. `exporter.py` 累积所有帧结果，最后保存为 JSON 或 CSV

## 输出格式

**JSON**（完整数据）：metadata + frames[]，每帧含 blendshapes、landmarks_3d、transformation_matrix
**CSV**（简洁）：每行一帧，列为 frame, timestamp_ms, 52 个 blendshape 值。兼容 Blender Faceit/ShapeKeyGen 插件

## CUDA 支持

MediaPipe 推理仅 CPU（TFLite XNNPACK）。未来 CUDA 加速路径：替换 OpenCV 视频 I/O 为 `cv2.cuda` 变体，detector 接口不变。
