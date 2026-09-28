# GR00T Visual / Prefill 双案例：实战方案与资源清单

两案例均为不编号的 L7 补充实战材料。Visual 练习权重生命周期取证，Prefill 练习量化消融与混精实验；共同连接 L1/L2/L3/L7/L8/L9。未覆盖的 L0、L4、L5、L6 基础按主课程继续，不把案例完成等同于全部主题掌握。

## 一、每次怎样练

每次约 30～45 分钟，视环境运行时间调整：先讲本实验必需的概念，读 20～50 行关键代码，再由学习者预测一项输出、执行一个实验、填写证据记录。导师在原案例文档批改结论与实验设计，必要时给参考答案，并区分“已阅读”“辅助完成”“独立复现”。

不要求手写完整量化框架；重点训练读懂入口、改变一个因素、检查真实生效状态、根据输出决定下一步。

## 二、三档实操条件

| 档位 | 能做什么 | 所需资源 |
|---|---|---|
| 离线读证据 | 比较 Tensor、检查参数、分析实验有效性 | 配对 NPY/NPZ、TSV、少量 checkpoint 参数、相关源码；通常 CPU 足够 |
| GPU 复现 | 加载探针、FX/PTQ 消融、混精对照、重新导出 | 可运行的原容器、完整依赖、模型与配置、GPU；容量按原环境实测 |
| 设备验收 | 验证导出、性能和真实任务效果 | HMONNX 推理器/编译器、目标 NPU、业务仿真或评测脚本、固定评测集 |

优先用已有服务器上的目录。大型 checkpoint、容器与 ONNX external data 无需复制进课程仓库；先提供准确路径和版本。访问服务器须使用已有授权连接，课程中不保存密码、令牌等凭据。

## 三、已确认在本地的材料

- `example_gr00t/` 下 11 个探针脚本，包括 visual 加载、导出、initializer 对比与 prefill FX/PTQ 消融。
- 两份用户提供的排查记录；已分别整理进 Visual 与 Prefill 案例课。
- 当前 prefill 消融脚本依赖外部 `quant_pipeline.py`，且为基线 QuantScheme 版本，缺少方案 B 的 helper 引用。仅这些脚本不足以直接重跑完整实验。

未检查到的资源在下面记为“待提供/确认”，不代表服务器上不存在。

## 四、最先补齐的最小材料包

| 材料 | 需要的具体内容 | 用途 |
|---|---|---|
| 当前工程入口 | `ptq.py`、`quant_pipeline.py` 及它们依赖的本地 helper | 还原 dump、dummy、wrap 和量化调用链 |
| Prefill converter/wrap | `groot/_model.py`、`qwen3_converter.py`、`qwen3_convert_config.py`，附实际根目录 | 追踪 Attention、校准输入和配置入口 |
| B 的实际修改 | `_bak_schemeB/` 基线或 Git diff；当前消融脚本；`build_prefill_quant_scheme` 等 helper | 重现真实试验而非只看记录 |
| 基线与 B 日志 | 完整消融输出、四路 TSV、算子统计与有效配置报告 | 判断指标、配置和产物是否属于同一轮 |
| 配对 Tensor | 至少一份固定 prefill 输入，wrap、no-PTQ、PTQ 基线、B 输出；Visual 修前/修后对应输入输出 | CPU 上开始比较、练习定位 |
| 运行说明 | 工作目录、容器名或镜像标识、Python 路径、PYTHONPATH、成功命令、代码 commit 与本地 diff | 排除包路径或版本串用 |

若上述资料很大，提供所在服务器及绝对路径即可。先有源码+日志+配对 Tensor，就可以开展第一档实验。

## 五、Visual 需要补充的专项资源

1. 修前/修后 `modeling_siglip2.py` 或精确 diff；`gr00t_n1d6.py`、`eagle_backbone.py`、AutoModel 注册与 policy 加载入口。
2. 两个实际 Transformers 环境的版本、安装路径及相关加载/初始化源码快照，避免把版本结论外推到其他安装。
3. checkpoint 的 config、索引与路径；离线阶段可仅导出选定 LayerNorm、FC1、patch embedding 参数及键名映射。
4. 失败/正常 HMONNX 及相应 external data，或离线阶段提供带节点输入名的 initializer 抽取文件。
5. `add_53`、post-LN、首层 LN 等中间 golden，注明模型版本和输入；以及加载前后探针原始日志。
6. 每套四路 dump 的来源：模型什么时候加载、是否复用对象、何时导出、权重摘要、输入哈希。

第 6 项很关键：若 native/FX 已坏却彼此接近，而 HMONNX 又与 FX 差异很大，仅凭“加载后初始化”不能完整解释这张比较表。需要检查不同加载实例、旧产物或运行路径差异，把每个结果的来源补齐。

## 六、Prefill 需要补充的专项资源

1. 方案 A 基线与修改版代码或 `_bak_schemeA/` 加对应 diff，用于审查合成 embedding 构造。
2. 精确版本的 xhquant：PTQ 入口、QuantScheme 规则匹配、QMaskedSoftmax、QGroupMatMul、QLinear 及 FakeQuant 实现；优先使用原容器，必要时提供相关源码范围。
3. 真实轨迹的校准 embedding 与独立评测 embedding，连同 mask、positions、图像/文字槽位、processor/config 和样本来源。来自同一轨迹的近邻帧应避免跨校准/评测泄漏。
4. 第 0、7、15 层匹配的中间输出，或可添加 hook/FX 节点记录的执行入口。
5. 新混精 HMONNX、编译日志、设备测量结果与任务评测入口。用户当前记录尚未包含该轮新导出和部署验收。

用于教学起步可先抽取 10～20 个代表性样本；这只是熟悉流程的小样本集，不是正式业务验收规模。

## 七、双方分工与后续交付

用户提供可访问的源码/数据路径、版本与现有成功运行命令。导师基于真实接口整理统一命令入口、分步实验说明、配对 Tensor 比较工具、权重抽查工具、量化配置生效报告和结果模板。

以上工具属于材料齐备后的实施内容；当前已交付的是案例、实验任务与资源清单，未声称提供完整可执行复现包。对专有框架不编造缺失 API。

每轮实验至少记录：案例与实验 ID、代码版本、环境路径、checkpoint 标识、校准/评测样本标识、实际精度配置、输入和产物哈希、命令、日志、结论与未验证项。保存可复现的实际输入比只记随机种子更可靠。

## 八、课程入口

- [Visual：加载与参数取证](AI_NPU系统课程_L7_GR00T_Visual精度Debug实战.md)
- [Prefill：PTQ 消融与混精](AI_NPU系统课程_L7_GR00T_Prefill精度Debug实战.md)

两案例暂存，不改变当前第 08 课 RoPE 的交互学习位置。
