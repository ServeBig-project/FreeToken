# Flash-Next 本地权重与容量证据

日期：2026-10-09。仅读取配置、索引和safetensors头，未加载GPU、未运行转换或模型；以下容量是元数据计算，性能和实际增槽尚未验证。

## 1. 找到的源

- 路径：`/data1/yuchen/models/Qwen3.8-Flash-Next-NVFP4`。
- 来源：本地README／qualification-notes标明 `RadixArk/Qwen3.8-Flash-Next-NVFP4`。
- 本地 `NVFP4_REVISION` 及config下载元数据共同记录 revision `7b719225242aacd3dbd3f9407468c2ee9a9d2594`。这是已有版本标识；未生成额外校验值。
- 容器：safetensors。专家 `weight` 为U8打包NVFP4，`weight_scale`为F8_E4M3，`weight_scale_2`为F32；PLE表为F8_E4M3，另有scalar scale。
- 实读的GDN、QSA、共享专家、HC投影和lm_head张量头为BF16。此目录不是已经dense FP8的交付产物；显式FP8策略需要转换这些源矩阵。
- 旁边有GGUF目录，不作为本任务选定的权重来源。不能把文件容器、专家量化、dense量化和KV编码混为一项。

## 2. Dense矩阵载荷

统计条件：文本塔、BF16、二维weight；不含MTP、视觉、路由专家和PLE大表。

| 角色 | 源BF16字节 |
| --- | ---: |
| GDN投影 | 4,170,055,680 |
| QSA投影 | 1,234,698,240 |
| HC残差混合 | 1,279,262,720 |
| 共享专家投影 | 471,859,200 |
| PLE投影 | 65,536,000 |
| lm_head | 1,271,398,400 |
| token embedding（不在本FP8 运行配置内） | 1,271,398,400 |
| 路由gate（不在本FP8 运行配置内） | 125,829,120 |
| 共享专家选择gate（不在本FP8 运行配置内） | 245,760 |

前六项的原始载荷从2 B降到1 B，节省约3.954773 GiB。最终净值还要计FP32行scale、投影融合／填充、后端实际工作区；不能将其当作实测显存差。

H=2560、I=640时，当前native NVFP4专家槽：

```text
2I*(H/2 + H/16 + 2) + H*(I/2 + I/16 + 2) = 2,772,480 B
```

因此上述载荷理论上约对应1531个native槽，扣scale等后约1530量级。repack后四个bank的载荷＋block scale约2,764,800 B／槽，alpha等另计。用户提出的1700槽目前没有同一基线下的实测支撑，必须用实际分配差核对，不能硬写进allocator。

## 3. KV与索引

12个QSA层、2个KV head、head_dim=256。K与V分别编码；INT8采用每token每head的BF16 scale。

```text
BF16 K/V：12 * 2 * 2 * 256 * 2 = 24,576 B/token
INT8 K/V＋scale：12 * (2 * 2 * 256 + 2 * 2 * 2) = 12,384 B/token
GPU压缩索引：12 * 128 * 2 / 4 = 768 B/token（按完整组取整）
```

| 历史token | BF16 K/V | INT8 K/V＋scale | 压缩索引 | KV理论节省对应native槽 |
| --- | ---: | ---: | ---: | ---: |
| 131072 | 3.000 GiB | 1.511719 GiB | 96 MiB | 576 |
| 157000 | 3.593445 GiB | 1.810759 GiB | 114.990 MiB | 690 |

570槽与128K规模的这个计算接近；它不能不带上下文长度就用于其他配置。表中没有计物理块浪费、尾页、请求状态、选中数据暂存或Graph。

## 4. 必须分开的验收结论

1. 算术节省：由真实载荷和scale推导。
2. 实际GPU／CPU占用：包含块粒度、临时／固定工作区及副本。
3. 真正增槽：新的资源方案是否把净余量分给专家；runtime总预算未减少时不能再算一份专家收益。
4. 质量与性能：dense FP8／INT8量化误差、157K任务质量、主机稀疏读取与专家搬运的PCIe竞争，均需独立测量。

本记录不沿用检查点README的GSM8K/AIME成绩为新增dense FP8或INT8 KV背书；这些新精度路径需要自己的验收。
