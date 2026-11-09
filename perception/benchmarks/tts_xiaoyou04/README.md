小优 04「明朗利落」的榜单接入代码。当前绑定累计 25,832 步的诊断候选，用于 JP6 试评。模型、声线、前端和验收报告均在固定版本的 JuiceFS 包中，逐文件校验。

首轮平台评测在首次 `config` 调用失败：服务加载默认 `tts` 和完整产品配置，评测器却将 `TTS_PLUGIN=tts_xiaoyou04` 作为 MCP 工具名使用。该次运行尚未合成音频，不能解读为模型效果失败。本版让仓库默认配置选择 04，并为评测器工具名提供别名；服务不依赖平台转发自定义环境变量。日志中的四个空云端配置字段按无操作处理，非空值仍明确拒绝。

该候选的 `normal_text_parity_passed` 和 `listening_approved` 均为 `false`。12 条完整波形的原有严格数值门槛尚未通过，部分相位边界和变速差异仍需处理；诊断报告随包保留。加载需显式使用 `XIAOYOU04_ALLOW_DIAGNOSTIC=1` 或配置中的 `allow_diagnostic_trial: true`；本诊断分支的默认配置明确启用后者。本轮平台评测用于获得真实链路和听感反馈，不表示正式模型验收通过。

诊断图的训练权重保持 FP32；声源相位累加和模拟原生 FMA 的临时计算采用 FP64。没有替换为公共底模或其他声音。JP6 单实例 token GPU 烟测已完成，三语各两轮、预热 RTF 0.20–0.23；这些数据不包含前端、MCP/ROS 或三模型并发，不能代替端到端评测。

代码从 `4paradigm/phanthymotus` 最新主干建立独立分支，保留公共 `tts` 工具及现有 ROS 音频协议。参考：[TTS 接口](https://github.com/4paradigm/phanthymotus/blob/main/perception/plugins/tts.py)、[感知入口](https://github.com/4paradigm/phanthymotus/blob/main/perception/main.py)。

已准备的接入行为：

- `MCP_PORT`、`WS_PORT` 优先于 YAML；空环境变量回退到 YAML，再使用原有端口默认值。
- `TTS_PLUGIN` 优先选择实现模块；为空时由 `plugins.tts.plugin` 选择，未指定时使用原有 `tts`。本分支的默认配置选择 `tts_xiaoyou04`。公开工具仍为 `tts`，同时支持评测器调用 `tts_xiaoyou04`。
- 新插件复用原有 `start`、`stop`、`speak`、`info`、`config`、`interrupt` 和多实例生命周期；语言为中文、英文、混读，固定一个 04 声线，仅暴露语速设置。
- 默认 `perception/config.yaml` 与专用 `plugins/xiaoyou04/config.yaml` 一致，只启用 TTS；配置明确选择独立前端 Python 和诊断包，不依赖 `CONFIG_PATH` 或前端解释器环境变量的转发。
- 模型按 archive SHA256 使用独立缓存目录，通过共享下载器加锁下载；逐文件核对大小和 SHA256，并验证声线形状、前端和词表。模型未绑定、文件损坏或请求 GPU 却没有 CUDA provider 时明确报错。
- 音素前端在独立 Python 3.11.15 子进程运行，保持 Misaki 0.9.4、spaCy 3.8.16 等训练依赖。JP5 的 ROS Python 3.8 与 JP6 的 ROS Python 3.10 均不需要更换。Docker 中只在绑定模型后安装该环境，准备分支跳过这一步。
- 原生 ONNX 的 token、固定声线、speed 输入由独立 ORT worker 处理。24 kHz FLOAT 波形沿用上游滤波器转换为 16 kHz PCM16；已有 ROS 节点负责音频帧和 EOF，`/tts/test` 仍包装为 WAV。

当前验证与剩余工作：

| 项目 | 状态 |
|---|---|
| 本地端口、插件选择、生命周期、PCM 分块、模型校验与原有 TTS 回归 | 已验证，属于设备外测试 |
| 真实训练前端的子进程传输 | 72 条对照：61 条音素及 token 一致，11 条原有拒绝一致；仍保留原有数字、单位和读音问题 |
| 冷缓存下载、断网复用、重启、缓存损坏拒绝、独立 ORT worker | 已用小型合成 ONNX 测试图验证，不是训练模型的推理效果证明 |
| 04 最终模型、ONNX 导出和正常文本等价核验 | 已绑定诊断快照；严格完整波形验收未通过，最终模型待训练、修订与验收 |
| 绑定固定模型、JuiceFS 上传和 HTTP 回下载校验 | 诊断快照已发布，完整 HTTP 回下载大小和 SHA256 一致 |
| JP5/JP6 真实前端、CUDA worker、MCP/ROS 单实例链路 | 已用现有镜像和本版源码复测，清除自定义环境变量，并重放平台工具名和空配置 |
| Orin 8G 上三个独立模型实例、首字延迟与 RTF | 未验证；需实测，不能以多张画布卡片代替 |
| public fork 和提交配置 | 首轮已提交但接入失败；修复版使用新 commit，保留原运行身份 |

模型发布包采用平铺目录，至少包含：

```text
model.onnx           # tokens(int64 [1,T]), style(float32 [1,256]), speed(float32 [1]) -> audio
voice.npy            # 唯一的 04 FP32 声线，[1,256]，禁止 pickle
config.json          # 与训练一致的音素词表
frontend.py          # 从对应训练前端导出的独立版本，只读同目录 config.json
vendor/              # 固定 en_core_web_sm 等前端模型数据
export_identity.json # 04 检查点 SHA、导出文件 SHA 和正常文本等价核验结果
```

为适配 JP5 所用 ORT，打包器要求 ONNX IR ≤ 8、opset ≤ 17。这只限制文件格式；具体算子、音质、内存和延迟仍须在两版设备上验证。ONNX 导出须使用训练后的 04 权重，保留原生声学路径和匹配前端；已有公共 Kokoro ONNX 不能代替训练产物。默认打包仍要求正常文本等价核验通过。用户要求的诊断评测可使用 `--diagnostic-trial`：必须明确保存 `normal_text_parity_passed: false`、`listening_approved: false` 和与模型/检查点对应的 `validation_report.json` 及 SHA256；运行时还需显式允许诊断候选。该入口不修改权重，也不把失败写为通过。

`export_identity.json` 中需包含 `voice: "04"`、`architecture: "Kokoro82M_v1.0"`、`precision: "FP32"`、`sample_rate: 24000`、`source_checkpoint_sha256`、`normal_text_parity_passed: true`，以及 `export_sha256` 中上述四个必需文件的真实哈希。核验报告应另行保存原文、音素、时长、声学比较和使用的随机源，不能只写一个通过标记。

制品在 Git 外打包，示例命令中的路径由实际版本替换：

```bash
python3 perception/tools/pack_xiaoyou04_release.py \
  --source /path/to/verified-onnx-export \
  --archive /path/to/releases/xiaoyou04-VERSION.tar.gz \
  --descriptor /path/to/releases/xiaoyou04-VERSION.json \
  --url http://172.28.4.81:34567/lizhuoju/embodied-ai/tts/xiaoyou04/VERSION/xiaoyou04-VERSION.tar.gz
```

打包本身不会上传。上传与完整 HTTP 回下载校验完成后，才把真实 descriptor 绑定到 `plugins/xiaoyou04/release.json`。权重、前端模型数据和大于 1 MB 的文件均留在模型存储，不进入 Git。绑定后的 Docker 构建将按 `deploy/setup_xiaoyou04_frontend.py` 安装独立前端运行时，发布包内的英文模型数据由 worker 加入模块搜索路径；运行时不自动联网补齐前端模型。

设备验证从单实例开始。设备版本、可用内存与端口确认后，拉取确切代码并使用本次个人测试目录；本地准备代码，测试盒只执行构建与测试。

```bash
python3 perception/tools/benchmark_xiaoyou04.py \
  --cases perception/benchmarks/tts_xiaoyou04/smoke.jsonl \
  --output /path/to/results/JP-VERSION-single --instances 1 --device gpu

python3 perception/tools/benchmark_xiaoyou04.py \
  --cases perception/benchmarks/tts_xiaoyou04/smoke.jsonl \
  --output /path/to/results/JP-VERSION-three --instances 3 --device gpu
```

三实例模式创建三个独立模型进程，待各自加载和预热后再同时发起请求，保存每条 WAV、首段耗时、非播放节奏下的 RTF、失败及模型身份。报告中的父进程 RSS 不包含前端和 ORT 子进程；容量验证还须记录整机/容器内存与 GPU 轨迹。

以上测量只覆盖 runtime。还需在两个 JP 上通过真实 MCP `tools/list`、`tools/call` 和 ROS 文本输入/音频输出回放，验证节点启动、重复启动、打断、停止、EOF 和多实例 topic 对应，并记录端到端首音延迟。现有模型按句段生成后流出 PCM，单句内部不是逐帧生成；运行时第一段速度不能代替完整链路的 1 秒准入结果。

代码、模型和双 JP 验证完成后，推送到 public 个人 fork，确认完整 commit 在远端可获取，再从 [提交模板](submission.example.yaml) 生成真实 `submit.yaml`。用户已明确本次榜单选择 JP6，配置固定为 `JP_VERSION: "6"`。最终由用户手动提交，准备过程不创建平台方案或发起上游 PR。
