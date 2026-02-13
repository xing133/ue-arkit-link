# Expression Detect

人脸表情检测项目：输入人脸视频 → 检测 52 个 ARKit blendshapes + 468 个 3D 面部关键点 → 输出标注视频 + 数据文件供 Blender 导入，或通过 Live Link 实时流式传输到 UE5 MetaHuman。

## 常用命令

### 离线处理模式

```bash
uv sync                                          # 安装依赖
uv run python -m src.main <video.mp4>             # 运行（带实时预览）
uv run python -m src.main <video.mp4> --no-preview # 无预览（headless）
uv run python -m src.main <video.mp4> --show-depth-map  # 带深度图侧边栏
uv run python -m src.main <video.mp4> --export-format csv  # CSV 导出
```

### 实时流式传输模式（UE5 Live Link）

```bash
uv run python -m src.main --stream --camera 0                    # 摄像头 → UE5
uv run python -m src.main --stream video.mp4                     # 视频文件流式传输
uv run python -m src.main --stream --camera 0 --livelink-host 192.168.1.100  # 远程 UE5
uv run python -m src.main --stream --camera 0 --no-preview       # 无预览（性能优化）
```

## 项目结构

```
src/
├── main.py           # CLI 入口，argparse 参数解析在 config.py
├── config.py         # PipelineConfig dataclass + parse_args()
├── pipeline.py       # 离线流程：视频读取 → 检测 → 可视化 → 导出
├── stream.py         # 流式流程：摄像头/视频 → 检测 → Live Link UDP → UE5
├── detector.py       # MediaPipe FaceLandmarker 封装（VIDEO 模式）
├── visualizer.py     # 绘制面部网格、blendshape 文本叠加、深度图
├── exporter.py       # JSON/CSV 导出，52 个 ARKit blendshape 名称定义在此
├── livelink.py       # LiveLinkSender，通过 UDP 发送 blendshapes 到 UE5
├── model_manager.py  # 模型自动下载（首次运行）
└── types.py          # FrameResult, VideoMetadata 数据结构
scripts/
└── blender_import.py # Blender 端导入脚本（在 Blender 脚本编辑器中运行）
models/               # 模型文件 face_landmarker_v2_with_blendshapes.task (~3.8MB)
output/               # 默认输出目录（_annotated.mp4 + _blendshapes.json）
docs/
└── architecture.md   # 完整技术文档（包含 UE5 Live Link 配置指南）
```

## 技术栈

- **Python 3.12+**, 包管理用 **uv**
- **MediaPipe 0.10.32** — FaceLandmarker Task API
- **OpenCV** — 视频读写 + 可视化绘制
- **NumPy** — 数组操作
- **PyLiveLinkFace** — UE5 Live Link Face UDP 协议实现

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

### 离线处理模式

1. `pipeline.py` 用 `cv2.VideoCapture` 逐帧读取视频
2. `detector.py` 每帧做 BGR→RGB 转换，调用 `FaceLandmarker.detect_for_video()`（timestamp 必须单调递增）
3. 返回 `FrameResult`：landmarks (468×3), blendshapes (52 个 name→score), transformation_matrix (4×4)
4. `visualizer.py` 在帧上绘制网格 + 文字叠加
5. `exporter.py` 累积所有帧结果，最后保存为 JSON 或 CSV

### 实时流式传输模式

1. `stream.py` 用 `cv2.VideoCapture` 打开摄像头或视频文件
2. `detector.py` 每帧检测，返回 `FrameResult`
3. `livelink.py` 的 `LiveLinkSender` 提取 blendshapes + 头部旋转，通过 UDP 发送到 UE5
4. `visualizer.py` 可选本地预览（`--no-preview` 可关闭以提升性能）
5. 无文件输出，数据直接流式传输到 UE5 MetaHuman

## 输出格式

**JSON**（完整数据）：metadata + frames[]，每帧含 blendshapes、landmarks_3d、transformation_matrix
**CSV**（简洁）：每行一帧，列为 frame, timestamp_ms, 52 个 blendshape 值。兼容 Blender Faceit/ShapeKeyGen 插件
**Live Link UDP**（实时流式）：通过 UDP 端口 11111 发送 52 个 ARKit blendshapes + 头部旋转到 UE5

## UE5 Live Link 接收端配置

### 快速开始

1. **启用插件**：Edit → Plugins → 搜索 "Live Link"，确认已启用
2. **打开 Live Link 面板**：Window → Live Link
3. **启动 Python 流式传输**：
   ```bash
   uv run python -m src.main --stream --camera 0
   ```
4. **绑定 MetaHuman**：
   - 选中关卡中的 MetaHuman Actor
   - Details → Face → Live Link Subject Name → 填入 `PythonFace`
   - 确保 Animation Mode 设为 Live Link

### 常见问题

- **看不到 Subject**：检查防火墙，确认 UDP 11111 端口未被占用
- **Subject 显示黄色/红色**：检查 Python 端日志，确认 MediaPipe 检测正常
- **MetaHuman 不动**：确认 Live Link Subject Name 与 `--livelink-subject` 参数一致（默认 `PythonFace`）
- **延迟明显**：使用本地 127.0.0.1，关闭预览窗口 `--no-preview`

详细配置步骤和故障排查见 `docs/architecture.md` 第 9 章。

## CUDA 支持

MediaPipe 推理仅 CPU（TFLite XNNPACK）。未来 CUDA 加速路径：替换 OpenCV 视频 I/O 为 `cv2.cuda` 变体，detector 接口不变。
