# 模型部署模式 ：镜像装代码，卷装模型

本文描述 Delveta 三个模型服务的部署方式：Embedding、cap-router（LayaChoice）和
tool-intent（Ollama）。核心思路是 **应用镜像只承载代码、运行时和依赖，模型权重放在
Docker 命名卷中**，由容器启动时的初始化步骤负责下载、校验和发布。

**状态标记**

| 标记 | 含义 |
|---|---|
| **[已实现]** | 代码与配置已落地。 |
| **[待实现]** | 已确认需要，但代码尚未完成。 |
| **[待验收]** | 代码已完成，但缺少本地单机以外的验证，或验证场景不完整。 |

只有同时满足“已实现”和“已验收”的项才被视为完成。文档末尾的第 12 节逐项列出现状。

---

## 1. 概述

三个服务镜像与卷的关系如下。图中的卷名是 Docker 中的实际名称，前缀来自
`docker-compose.yml` 顶层的 `name: delveta`。

```
                       docker compose (project: delveta)
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
   embedding                   cap-router               tool-intent-model
   delveta-embedding-int8      delveta-cap-router       delveta-tool-intent-init
        │                           │                    （一次性初始化）
        │ /model                    │ /models/layachoice         │ /root/.ollama
        ▼                           ▼                            ▼
 delveta_embedding-model     delveta_laya-model       delveta_toolintent-cache
                                                                  ▲
                                                                  │ 同一命名卷
                                                            tool-intent
                                                          delveta-tool-intent
```

| 服务 | 职责 | 模型体量 |
|---|---|---|
| `embedding` | INT8 ONNX 文本向量服务，兼容 `POST /embed` | 约 586 MB |
| `cap-router` | LayaChoice 能力选择模型，`POST /v1/systemone` | 约 678 MB |
| `tool-intent-model` + `tool-intent` | 参数抽取模型（GGUF，Ollama 服务） | 约 444 MB |

每个服务只加载自己声明过的文件。仓库中的训练代码、README、其他量化版本不会被下载。

## 2. 架构原则

1. **镜像不装权重。** 应用镜像只包含代码、运行时和依赖。模型与镜像解耦，镜像体积
   不随模型增长。**[已实现]**
2. **模型放在 Docker 命名卷，不使用宿主机绝对路径。** 部署不依赖盘符、项目目录或
   预先存在的模型文件，换机器只需 Docker 和首次下载所需的网络。**[已实现]**
3. **首次启动自动下载、校验、持久化；后续启动直接复用。** 复用判定基于发布回执
   与文件长度，不重复计算全量 SHA256。**[已实现]**
4. **下载逻辑只有一份。** 断点续传、完整性校验、并发锁、原子发布都实现在
   `deploy/model_store`；仓库、版本和文件清单由各服务的运行时配置提供。**[已实现]**
5. **模型版本与代码版本相互独立。** 三个服务的模型参数都在运行时注入：tool-intent
   使用 `TOOL_INTENT_MODEL_*` 环境变量；embedding 与 cap-router 使用
   `MODEL_SPEC_JSON`，由 `deploy/model_config/*.env` 通过 Compose `env_file` 注入。
   换模型只需修改配置并重建容器，不需要重建镜像。**[已实现]**
6. **提供可选的离线导出/导入。** 用于在无法访问 Hugging Face 的环境中迁移模型。
   **[已实现]**；跨机器验收 **[待验收]**。
7. **Ollama 模型只有在注册并验证成功后才被标记为就绪。** 服务在模型未注册时不会
   报告健康，也不会绑定公开端口。**[已实现]**

## 3. 卷内目录结构

### 3.1 embedding 与 cap-router

两者的卷布局相同，只有挂载点和加载路径不同。

```
<volume>/
├── versions/
│   └── <revision>/              # 不可变快照，按 commit SHA 命名
│       └── ...                  # 保持仓库内的相对路径
├── current -> versions/<revision>
├── .delveta-model.json          # 发布回执：repo、revision、每个文件的 size/sha256
├── .fetch.lock                  # 跨进程初始化锁
└── .incoming-<revision>/        # 下载暂存；发布后不存在
```

| 条目 | 作用 | 生命周期 |
|---|---|---|
| `versions/<revision>/` | 已校验的模型快照 | 保留当前版本；发布新版本时删除其他版本 |
| `current` | 服务实际读取的路径（符号链接） | 通过临时链接加 `os.replace` 原子切换 |
| `.delveta-model.json` | 复用判定的依据；与 pin 完全一致时跳过下载 | 每次发布后重写 |
| `.fetch.lock` | 保证同一卷只被一个进程初始化 | 常驻，不参与导出 |
| `.incoming-<revision>/` | 未完成的下载；按 revision 命名以支持续传 | 发布后重命名；其他 revision 的暂存会被清理 |

### 3.2 tool-intent

Ollama 的卷根目录是 `/root/.ollama`，其中同时存在暂存区、标记文件和 Ollama 自己的
模型库。

```
delveta_toolintent-cache/
├── .model-store/                # 初始化服务的暂存区（注册成功后删除）
│   ├── versions/<revision>/
│   └── current -> versions/<revision>
├── .model-ready.json            # 标记：这些字节已注册到 Ollama
├── models/
│   ├── blobs/sha256-<digest>    # 权重唯一持久副本；digest 等于 GGUF 的 SHA256
│   └── manifests/registry.ollama.ai/library/delveta-pe/q5km
└── .fetch.lock
```

`.model-store/` 是过渡目录：注册并校验成功后会被删除，Ollama 的 blob store 成为唯一
持久副本。导出卷时它可能不存在，这是正常状态。

## 4. 启动流程

### 4.1 embedding 与 cap-router

```
container start
   │
   ├─ 读取并校验 MODEL_SPEC_JSON（Compose env_file 注入）
   │     └─ 缺失、JSON 非法、revision 不是完整 commit SHA、路径或摘要非法 → 退出 1
   ├─ 获取 <volume>/.fetch.lock（阻塞等待）
   ├─ 回执与 pin 一致 且 current 中文件长度正确？ ── 是 ──► 直接 exec 服务
   │                                              否
   ├─ 下载缺失文件到 .incoming-<revision>/
   │     └─ 每个文件从现有长度开始，使用 HTTP Range 续传
   ├─ 校验每个文件的 size 与 sha256
   │     └─ 长度正确但摘要错误：清空后重下一次；再失败则退出
   ├─ 发布：versions/<revision> ← 暂存目录
   │        原子切换 current
   │        写回执，删除其他版本
   └─ exec 服务进程（uvicorn / python server.py）
```

任何一步失败都以非零码退出，服务进程不会启动，端口不会被绑定。系统没有降级模型。

配置在每次容器启动时重新读取，因此修改配置后重建容器即可生效；镜像内容不变。

### 4.2 tool-intent

`tool-intent` 是可选服务。只有显式启动 `tool-intent` 及其初始化服务时才走以下
流程；启动脚本默认只拉起 `embedding` 与 `cap-router`，不启动参数抽取器。

模型下载与模型注册被拆成两个容器，Compose 用
`depends_on: condition: service_completed_successfully` 串联。

```
docker compose up -d tool-intent
   │
   ├─ tool-intent-model（一次性）
   │    1. .model-ready.json 与 pin 一致？ ── 是 ──► 退出 0
   │    2. Ollama manifest 中的模型层 digest 等于 pin？ ── 是 ──► 补写标记，退出 0
   │    3. 下载并校验 GGUF 到 .model-store/，退出 0
   │
   └─ 仅在初始化服务退出 0 后：
        tool-intent
          1. 启动一个仅用于注册的 Ollama 进程
          2. 已注册模型的 blob digest 等于 pin？
               是 → 写标记，删除 .model-store，exec ollama serve
               否 → 从 .model-store/current/<file> 执行 ollama create
                    重新校验 blob digest
                    写标记，删除 .model-store，exec ollama serve
          3. 模型未注册且 .model-store 不存在
               → 删除标记，退出 1（下次启动重新下载）
```

Ollama 会以文件的 SHA256 命名权重 blob，因此 `ollama show --modelfile` 中的
`FROM ...sha256-<digest>` 可以直接作为来源证明，而不只是“同名标签存在”。

健康检查只在模型已注册时通过：`ollama list | grep -q "$OLLAMA_MODEL"`。

> 注册只让权重可用，并不代表已经加载。服务启动后的第一个推理请求要把模型装入内存，
> 耗时明显高于后续请求。API 在启动时会在后台发起一次预热请求，使用独立的加载超时，
> 不受单次参数抽取超时限制；预热失败只记录日志，不影响启动。模型常驻由 sidecar 的
> `OLLAMA_KEEP_ALIVE` 设置保证。

### 4.3 复用判定的粒度

正常启动只比较回执与文件长度，不重新计算摘要。完整性由下载时的校验保证一次，且
`versions/<revision>` 发布后不再修改。这样避免了每次启动对 0.64 GB 文件重复哈希。

## 5. 设计理由

**5.1 为什么不把权重打进镜像**

镜像会被推送到仓库、被每台机器拉取，并在每次构建时产生新层。权重放进去会让镜像
体积达到 GB 级，构建和分发成本远高于下载一次模型；同时模型更新会迫使代码镜像也
产生新版本。

**5.2 为什么用命名卷而不是宿主机目录**

宿主机目录会把部署绑死到具体盘符和项目路径，换机器就要重新约定路径；命名卷由
Docker 管理，跨平台行为一致，并且可以随 Compose 项目一起创建和销毁。代价是卷不能
用文件管理器直接拷贝，因此提供导出/导入脚本。

**5.3 为什么原子发布**

服务只能看到完整的模型。下载写在与加载路径无关的暂存目录里，只有全部文件校验通过
后才通过 `current` 的原子切换暴露。任何中途失败都不会留下“看起来存在、实际不完整”
的加载路径。

**5.4 为什么使用固定暂存目录和 Range 续传**

暂存目录按 revision 命名。连接中断后，已写入的前缀就是下一次的起始偏移；容器被重建
时这些字节仍在卷里。下载器使用标准库 `urllib` 发起 Range 请求，不依赖会删除临时文件
的下载库。

**5.5 为什么同时固定 size 和 sha256**

只校验大小无法发现同长度的内容替换；只校验摘要则无法在下载过程中判断是否已完成。
两者同时固定，并在发布前逐文件校验，可以拒绝被静默替换的上游文件。

**5.6 为什么基础镜像按 digest 固定**

标签会移动，digest 不会。`python:3.11-slim`、`ollama/ollama` 和导出脚本使用的
`alpine` 都固定到具体 digest，保证不同时间、不同机器构建出相同的运行时。

**5.7 为什么 tool-intent 拆成两个容器**

Ollama 官方镜像没有 HTTP 客户端，也不应该为下载而在其中安装 curl/apt。拆开后：
下载逻辑复用 Python 的 `model_store`；Ollama 镜像保持原样，只负责注册和推理；
Compose 的依赖条件保证“字节未就绪就不启动服务”。

**5.8 为什么标记文件之外还要读 Ollama manifest**

升级前的卷可能已经注册了模型但没有标记文件。若只看标记，就会重复下载 444 MB。
读取 manifest 中的模型层 digest 可以在不信任标记的情况下确认字节来源，从而直接复用。

**5.9 为什么配置走 env_file 而不是挂载配置文件**

模型参数不属于镜像，也不应该要求额外的宿主机 bind mount。Compose 的 `env_file`
在创建容器时把配置注入为环境变量：镜像保持通用，配置随部署环境变化，容器内没有
新增挂载点，也不需要为了换模型重建镜像。结构化参数（仓库、revision、文件清单、
size、sha256、加载路径）放在单个 `MODEL_SPEC_JSON` 中，由启动代码一次性解析和校验，
避免十几个相互关联的环境变量出现半配置状态。

代价是 env 值在创建容器时固化：修改配置必须重建容器，`restart` 不会读取新值。这一点
在更新流程中显式说明。

**5.10 代价与边界**

- 首次启动需要网络和额外时间（下载加校验）。
- 卷只在当前 Docker 环境中有效；迁移需要导出/导入。
- 只保留当前版本，回滚到旧版本需要重新下载。
- 导入使用文件长度快速判定，不重新哈希（见第 12 节）。

## 6. 离线导出与导入

导出脚本把命名卷打包成一个 `tar.gz`。归档保留 `versions/`、`current` 符号链接和
回执，因此导入后的卷可以直接复用。

**重要：脚本接收的是 Docker 中的实际卷名，不是 Compose 中的卷键。** 本项目
Compose 项目名为 `delveta`，所以实际卷名为 `delveta_embedding-model`、
`delveta_laya-model`、`delveta_toolintent-cache`。

### 6.1 导出（源机器）

```bash
deploy/model_volume_bundle.sh export delveta_embedding-model embedding-model.tar.gz
deploy/model_volume_bundle.sh export delveta_laya-model laya-model.tar.gz
deploy/model_volume_bundle.sh export delveta_toolintent-cache toolintent-cache.tar.gz
```

- 归档已存在时脚本拒绝覆盖。
- 不包含 `.fetch.lock`、`.incoming-*`、`.current.tmp`。
- Windows 上需要 Git Bash；脚本会通过 `cygpath` 转换宿主路径。

### 6.2 导入（目标机器）

卷必须已存在。用 Compose 创建，不要用 `docker volume create`，否则
`docker compose down -v` 不会管理这个卷。

```bash
docker compose create embedding
deploy/model_volume_bundle.sh import delveta_embedding-model embedding-model.tar.gz
docker compose up -d embedding

docker compose create cap-router
deploy/model_volume_bundle.sh import delveta_laya-model laya-model.tar.gz
docker compose up -d cap-router

docker compose create tool-intent-model
deploy/model_volume_bundle.sh import delveta_toolintent-cache toolintent-cache.tar.gz
docker compose up -d tool-intent-model tool-intent
```

导入后的首次启动应命中回执并直接复用，不发起网络请求。

### 6.3 状态与验证

- 导出、导入、导入后复用：**[已实现]**。
- 本机两个独立卷之间的往返验证了归档会保留已发布的目录布局，导入后首次启动
  可直接复用，不发起下载。
- 跨真实机器（不同宿主机、不同架构）的导入：**[待验收]**。
- 导入时重新校验 sha256：**[待实现]**。当前快速路径只比较文件长度。

## 7. 日常操作

| 操作 | 命令 | 对模型卷的影响 |
|---|---|---|
| 默认启动（embedding + cap-router） | `docker compose up -d embedding cap-router` | 无则下载，有则复用 |
| 按需启动（追加参数抽取器） | `docker compose up -d tool-intent-model tool-intent` | GGUF 无则下载，有则复用 |
| 查看初始化日志 | `docker compose logs --tail=100 tool-intent-model` | 无 |
| 查看状态 | `docker compose ps` | 无 |
| 重启容器 | `docker compose restart embedding` | 保留；**不会**重新读取 env_file |
| 改模型配置 | 编辑 `deploy/model_config/*.env`，然后 `docker compose up -d --force-recreate <service>` | 保留旧版本，直到新版本校验并切换 |
| 改代码后重建镜像 | `docker compose build embedding` 然后 `up -d` | 保留，不重新下载 |
| 停止但保留卷 | `docker compose down` | 保留 |
| 停止并删除卷 | `docker compose down -v` | **删除**，下次启动重新下载 |
| 查看卷内容 | `docker compose exec embedding ls -la /model` | 只读 |
| 查看卷占用 | `docker system df -v` | 只读 |

需要注意的容器命名：

- Compose 服务名：`embedding`、`cap-router`、`tool-intent-model`、`tool-intent`
- 容器名：`delveta-embedding`、`delveta-cap-router`、`delveta-tool-intent`
  （`tool-intent-model` 是一次性容器，退出后保留为 Exited 状态）
- 卷的 Compose 键：`embedding-model`、`laya-model`、`toolintent-cache`
- 卷的实际名称：带 `delveta_` 前缀

## 8. 模型更新

### 8.1 tool-intent

所有参数都通过环境变量传入，修改后重建容器即可，不需要重建镜像：

```
TOOL_INTENT_MODEL_REPO
TOOL_INTENT_MODEL_REVISION
TOOL_INTENT_MODEL_FILE
TOOL_INTENT_MODEL_SHA256
TOOL_INTENT_MODEL_SIZE
OLLAMA_MODEL
```

```bash
docker compose up -d --force-recreate tool-intent-model tool-intent
```

### 8.2 embedding 与 cap-router

**配置文件位置（宿主机）：**

- `deploy/model_config/embedding.env`
- `deploy/model_config/layachoice.env`

Compose 通过 `env_file` 在创建容器时把文件内容注入为 `MODEL_SPEC_JSON`。这个值不写入
镜像，也不通过 bind mount 出现；容器内的 `bootstrap.py` 只从环境变量读取。

**更新步骤：**

1. 编辑对应的 `.env` 文件。保持双引号包裹和 `\"` 转义；修改 `revision`、`files`、
   `size`、`sha256`，必要时修改 `load_path`。
2. 重建容器：

```bash
docker compose up -d --force-recreate embedding
# 或
docker compose up -d --force-recreate cap-router
```

`up -d` 也能检测到环境变化并自动重建，显式加 `--force-recreate` 更明确。
**`docker compose restart` 不会读取新的 env 值**，因为它沿用已创建容器的环境。

整个过程不需要 `docker compose build`，也不需要重新打镜像。

**启动时的行为：**

1. 解析并校验 `MODEL_SPEC_JSON`；缺失或格式错误立即退出 1。
2. 回执与配置一致且文件长度正确 → 直接复用，不联网。
3. 否则下载到 `.incoming-<revision>/`（支持断点续传），逐文件校验 size 与 sha256。
4. 全部校验通过后才发布 `versions/<revision>`，并原子切换 `current`。
5. `current` 切换成功、回执写回之后，才删除其他版本目录。

**确认“新版本已发布后才清理旧版本”：**

```bash
docker compose logs embedding | grep -E 'spec:|verified|ready:|reusing'
docker compose exec embedding python -c "import os;print(os.readlink('/model/current'))"
docker compose exec embedding python -c "import os;print(sorted(os.listdir('/model/versions')))"
```

日志顺序应为：`spec:` → 每个文件的 `verified` → `ready: ... -> versions/<新revision>`。
如果没有出现 `ready:`，进程已经在校验阶段失败退出，`current` 仍指向旧版本，旧版本
目录也仍然存在。只有 `ready:` 出现后，其他版本才会被删除。

### 8.3 获取新版本的 pin

Hugging Face 的 tree API 同时返回文件大小和 LFS 对象 ID，后者就是 SHA256：

```bash
curl -s "https://huggingface.co/api/models/<repo>/tree/<revision>?recursive=true" \
  | jq -r '.[] | select(.type=="file") | [.path, .size, (.lfs.oid // "-")] | @tsv'
```

只把服务实际加载的文件写入清单，不要把整个仓库照搬进去。`load_path` 决定服务从
`current` 下的哪个子目录加载（embedding 为 `INT8`，cap-router 为 `.`）。

### 8.4 回滚

把 revision 改回旧值并重建容器（embedding 与 cap-router 使用
`up -d --force-recreate`，tool-intent 使用 `up -d --force-recreate tool-intent-model
tool-intent`）。由于只保留一个版本，旧版本需要重新下载；无法联网时使用事先导出的
归档。

## 9. 注意事项与排障

- `docker compose down -v` 会删除三个模型卷。这是唯一会自动触发重新下载的日常操作。
- `docker volume prune`、Docker 重置或卸载同样会删除卷。
- 导出/导入命令必须使用实际卷名（带 `delveta_` 前缀）。
- 归档按二进制传输；脚本不解压到宿主机，也不校验归档本身的完整性。
- embedding 与 cap-router 的配置由 `env_file` 注入；修改后必须重建容器，
  `restart` 不生效。
- `MODEL_SPEC_JSON` 缺失或格式非法时，初始化在校验阶段退出 1，不会改动 `current`。
- 镜像 digest 固定后，升级基础镜像需要显式修改 digest。
- `HF_ENDPOINT` 用于镜像站或代理；`HF_TOKEN` 用于私有仓库。二者都在 Compose
  环境中透传。
- `HF_HUB_DISABLE_XET=1` 保持开启：Xet 的分块缓存位于容器文件系统，容器重建后丢失。
- 只保留当前模型版本；回滚需要重新下载或使用归档。
- 下载前没有磁盘空间预检（见第 12 节）。

| 现象 | 原因 | 处理 |
|---|---|---|
| `model volume ... is not writable` | 卷权限异常 | 重建容器；初始化以 root 运行 |
| 反复出现 `resuming at N/M bytes` | 网络在文件完成前中断 | 等待重试，进度已保留 |
| `sha256 mismatch` 后出现一次完整重试 | 固定长度但内容损坏 | 自动处理；第二次失败说明 pin 或上游有问题 |
| `model ... is not registered and ... is absent` | 模型被删除，暂存目录也已删除 | 标记会自动移除；重新启动即可下载 |
| 服务健康但仍是旧模型 | 只执行了 `restart`，或配置未实际变化 | `up -d --force-recreate <service>`；确认 `revision` 已修改 |
| `MODEL_SPEC_JSON is not set` | 服务缺少 `env_file` 或变量被覆盖 | 检查 Compose 的 `env_file` 与变量优先级 |
| 只有某台机器能下载 | 代理或 `HF_ENDPOINT` 配置不同 | 统一 Compose 环境变量 |

## 10. 接入新模型

### 10.1 必需参数

| 参数 | 含义 | 示例 |
|---|---|---|
| `repo_id` | Hugging Face 仓库 | `eric-ml-nlp/Delveta-BGE-M3-v1` |
| `revision` | 不可变的 commit SHA，禁止使用分支或可移动标签 | `625d25d3…` |
| `files` | 服务启动必需的文件：路径、字节数、SHA256 | `INT8/model.onnx`, `568511234`, `0e0ada7f…` |
| `load_path` | `current` 下服务实际加载的子目录 | `INT8`（embedding）、`FP16`（cap-router） |
| 卷 | Compose 卷键与挂载点 | `embedding-model` → `/model` |
| 加载路径 | 服务读取的目录，由 `load_path` 决定 | `/model/current/INT8` |
| 健康检查 | 首个健康判定前的等待时间，需覆盖下载 | `start_period: 900s` |

### 10.2 接入步骤

1. 确认服务实际加载哪些文件，只列这些文件。
2. 从 HF tree API 取得每个文件的 `size` 与 `lfs.oid`（SHA256）。
3. 新建 `bootstrap.py`，调用 `load_model_spec()` 取得配置，再调用 `ensure_model()`。
4. Dockerfile 复制 `model_store` 与 `bootstrap.py`，入口设为 bootstrap，
   `CMD` 保留原服务命令。配置不复制进镜像。
5. 新建 `deploy/model_config/<service>.env`，用 `MODEL_SPEC_JSON` 保存参数；Compose
   中声明 `env_file` 与命名卷。模型权重只挂命名卷，配置不引入 bind mount。
6. 健康检查的 `start_period` 覆盖“下载 + 校验 + 加载”的最坏耗时。
7. 在空卷上验证首次下载，在已填充的卷上验证复用。
8. 如果需要离线部署，用导出/导入脚本做一次往返。

### 10.3 配置注入方式

模型参数既不在镜像内，也不通过 bind mount 提供。Compose 在创建容器时把 `env_file`
注入为 `MODEL_SPEC_JSON`，`bootstrap.py` 在启动时解析。因此同一镜像可以在不同环境
加载不同模型版本，配置内容也不会因为镜像分发而被固定。

更换配置的唯一要求是重建容器；`restart` 不生效。

## 11. 三个模型服务的配置

### 11.1 总览

| 项目 | embedding | cap-router | tool-intent |
|---|---|---|---|
| Compose 服务 | `embedding` | `cap-router` | `tool-intent-model` + `tool-intent` |
| 镜像 | `delveta-embedding-int8:latest` | `delveta-cap-router:latest` | `delveta-tool-intent-init:latest` + `delveta-tool-intent:latest` |
| 构建上下文 | `./deploy` | `./deploy` | `./deploy` |
| Dockerfile | `embedding_int8/Dockerfile` | `laya/Dockerfile` | `tool_intent/init.Dockerfile` + `tool_intent/Dockerfile` |
| 模型配置 | `deploy/model_config/embedding.env` | `deploy/model_config/layachoice.env` | 环境变量（Compose） |
| 卷（Compose 键） | `embedding-model` | `laya-model` | `toolintent-cache` |
| 卷（实际名称） | `delveta_embedding-model` | `delveta_laya-model` | `delveta_toolintent-cache` |
| 挂载点 | `/model` | `/models/layachoice` | `/root/.ollama` |
| 加载路径 | `/model/current/INT8` | `/models/layachoice/current/FP16` | `delveta-pe:q5km`（Ollama blob） |
| 仓库 | `eric-ml-nlp/Delveta-BGE-M3-v1` | `eric-ml-nlp/Delveta-LayaChoice-v2` | `eric-ml-nlp/Delveta-Qwen3-0.6B-PE-v1` |
| Revision | `625d25d3749076cee90abcd1904da3a6e7ce4c9b` | `24981b07e1efca2c15aafc165d5d1b888574f717` | `1a75945e12bfeb203910bdb765e7cdf07f6a1126` |
| 下载总量 | 585,611,532 字节（约 586 MB） | 678,202,185 字节（约 678 MB） | 444,414,752 字节（约 444 MB） |
| 宿主端口 | `18080 → 80` | `127.0.0.1:18092 → 8000` | `127.0.0.1:18091 → 11434` |
| 健康检查起点 | `start_period: 900s` | `start_period: 1800s` | `start_period: 1800s` |

### 11.2 环境变量

| 变量 | embedding | cap-router | tool-intent |
|---|---|---|---|
| `MODEL_SPEC_JSON`（来自 `env_file`） | ✓ | ✓ | |
| `TOOL_INTENT_MODEL_REPO` / `_REVISION` / `_FILE` / `_SHA256` / `_SIZE` | | | ✓ |
| `OLLAMA_MODEL` | | | ✓ |
| `HF_ENDPOINT` | ✓ | ✓ | ✓ |
| `HF_TOKEN` | ✓ | ✓ | ✓ |
| `HF_HUB_DISABLE_XET` | ✓ | ✓ | ✓ |
| `MODEL_DIR` | 由 bootstrap 按 `load_path` 设置 | | |
| `LAYA_CHOICE_MODEL_DIR` | | 由 bootstrap 按 `load_path` 设置 | |

### 11.3 基础镜像

| 用途 | 固定值 |
|---|---|
| embedding、cap-router、tool-intent 初始化 | `python:3.11-slim@sha256:6f31d6e9ba2b0a787a3f81c37b004155b87b9efa1b771182bd550c1615745be5` |
| tool-intent 服务 | `ollama/ollama@sha256:8262851b2846b87c649eddf3e76beb270c52f4d1bc94559f47efde16b0841551` |
| 卷导出/导入脚本 | `alpine@sha256:28bd5fe8b56d1bd048e5babf5b10710ebe0bae67db86916198a6eec434943f8b` |

### 11.4 tool-intent 的两个容器

| | `tool-intent-model` | `tool-intent` |
|---|---|---|
| 类型 | 一次性（`restart: "no"`） | 常驻（`restart: unless-stopped`） |
| 网络 | 需要，访问 Hugging Face | 不需要 |
| 职责 | 下载、校验、发布 GGUF | 注册模型、提供推理服务 |
| 成功标志 | 退出码 0 | 健康检查通过 |
| 共享 | `delveta_toolintent-cache` | `delveta_toolintent-cache` |

## 12. 落地状态

### 12.1 已实现（Implemented）

| 能力 | 状态 | 本地验证证据 |
|---|---|---|
| 镜像/卷分离，三个命名卷 | 已实现 | `docker compose config` 中模型服务只挂命名卷，无模型宿主路径 |
| 模型配置运行时注入 | 已实现 | 同一镜像仅修改 `revision` 配置后完成下载、发布与旧版本清理；镜像 ID 未变化 |
| 首次自动下载、校验、原子发布 | 已实现 | 三个模型的首次启动均完成下载、逐文件校验与发布 |
| 断点续传（HTTP Range） | 已实现 | 三个模型均出现 `resuming at` 并从偏移继续到完整长度 |
| 每文件 size + SHA256 pin | 已实现 | pin 来自 `MODEL_SPEC_JSON`；发布前逐文件输出 `verified` |
| 并发锁 | 已实现 | 通过 `fcntl.flock` 提供卷级互斥；双容器并发初始化尚未实测（见 12.3） |
| 基础镜像 digest 固定 | 已实现 | 四个镜像均按 digest 构建并运行 |
| tool-intent 独立初始化 | 已实现 | 初始化服务退出 0 后服务容器启动；生产迁移时未重复下载 |
| Ollama 注册来源校验 | 已实现 | blob digest 等于 GGUF SHA256；不匹配时重新注册 |
| 标记缺失时读取 manifest | 已实现 | 升级后的生产卷复用旧模型，未触发 444 MB 下载 |
| 失败自愈 | 已实现 | 删除模型后重启退出 1 并移除标记 |
| 卷导出/导入 | 已实现 | 同机往返验证了归档保留已发布布局；导入后首次启动直接复用，无下载 |
| 健康检查覆盖初始化 | 已实现 | 三个服务均为 healthy；embedding 与 cap-router 健康端点返回正常 |

### 12.2 待实现（Pending Implementation）

1. **导入时重新校验 SHA256。** 导入后走长度快速路径，归档在传输中损坏且长度不变
   时无法发现。需要增加显式 `--verify` 模式。
2. **`model_store` 的自动化测试。** 目前没有覆盖续传、锁、原子发布、损坏重试的
   单元或集成测试。
3. **下载前的磁盘空间预检。** 空间不足时会在写入过程中失败，缺少启动前的明确提示。

### 12.3 待验收（Pending Verification）

1. **跨真实机器导出/导入。** 目前只验证了同一台机器上的两个卷。
2. **初始化失败时 Compose 的拦截行为。** 已确认初始化进程退出 1；尚未验证
   `tool-intent` 在该情况下保持未启动。
3. **私有仓库、代理与镜像站。** `HF_TOKEN`、`HF_ENDPOINT` 的实际组合未验证。
4. **非 amd64 架构。** digest 只在本机 amd64 上解析过。
5. **Xet 开关对照。** 当前固定关闭，未做开启后的行为对照。
6. **同一卷的并发初始化。** 锁已实现，但尚未用两个容器同时初始化同一个空卷做实测。

以上三类状态分别对应第 2 节的架构原则；任何未通过验收的能力都不应对外表述为“已完成”。
