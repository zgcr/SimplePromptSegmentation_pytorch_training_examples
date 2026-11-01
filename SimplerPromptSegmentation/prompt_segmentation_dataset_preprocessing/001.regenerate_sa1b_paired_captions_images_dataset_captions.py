# coding:utf8
import os
import re
import json
import time
import random
from pathlib import Path
import base64
import cv2
import numpy as np
import requests
from tqdm import tqdm
from datetime import datetime
from typing import Tuple
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pycocotools import mask as mask_utils

os.environ['http_proxy'] = ""
os.environ['https_proxy'] = ""

# ✅ 避免每个进程内 opencv 抢占全部核心（多进程 + 多线程场景下必须限制）
cv2.setNumThreads(1)

# ====================== 数据集配置 ======================
DATASET_ROOT = '/root/autodl-tmp/interactive_segmentation_dataset'
# ✅ 要处理的子集列表, list 形式便于灵活指定
SUBSET_NAME_LIST = [
    'sa_000000',
]
SPLIT_NAME = 'train'

# ✅ 所有新文件都保存到独立目录, 原数据集保持只读
CAPTION_SAVE_ROOT = '/root/autodl-tmp/interactive_segmentation_dataset_captions'
GLOBAL_CAPTION_SUFFIX = '_global_caption.json'
OUTPUT_JSON_SUFFIX = '_deepseek_flash_output.json'

# ✅ mask 面积占整图面积的最小比例, 过滤过小的碎片 mask
MIN_MASK_AREA_RATIO = 0.01

# ====================== DeepSeek API 配置 ======================
# DeepSeek-V4.1-Flash, 模型名使用 deepseek-flash, 支持图像理解
DEEPSEEK_URL = 'https://api.deepseek.com/chat/completions'
DEEPSEEK_MODEL = 'deepseek-flash'
DEEPSEEK_API_KEY = ''

# ✅ 输入图片保持宽高比, 长边 resize 到 1024, 控制图像 token 消耗
IMAGE_LONG_SIDE = 1024
IMAGE_JPEG_QUALITY = 95
# low 会把图片缩到 512x512, high/original/auto 保留原图
IMAGE_DETAIL = 'high'

MAX_OUTPUT_TOKENS = 1024
REQUEST_TIMEOUT = 180
MAX_RETRY_TIMES = 3
RETRY_BACKOFF_SECONDS = 1.0

# ====================== 并发配置 ======================
# deepseek-flash 并发限制为 2500(按账号计), 这里用 进程数 x 每进程线程数 控制总并发
POOL_NUM = 10
THREAD_PER_PROC = 24

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

PROMPT_TEXT_GLOBAL = '''
# 🎯 任务：整张图像的全局描述（NON-THINKING）

## 🔒 视觉依据硬约束
- 只描述本张图片中【可直接看到的像素内容】。
- 禁止想象、推测、补全任何图片中看不到的对象。
- 禁止描述“轮廓”“剪影”“阴影占位”等无实际像素支撑的内容。

---

## ✅ 输出结构（必须严格遵守）
只允许输出【两句中文】，共一段纯文本。

- 第 1 句：概括全图整体场景，≤40 字，必须以中文句号“。”结尾。
- 第 2 句：以“画面中包含”开头，依次列出画面中主要的可指代对象名词，
  对象名词之间用中文逗号“，”分隔，≤60 字，必须以中文句号“。”结尾。

---

## ✅ 对象名词要求
- 必须具体、可指代（如：轿车、小孩、橱柜、水槽、招牌、路灯）。
- 禁止使用“物体”“东西”“若干物品”等笼统词。
- 只列出能从像素中直接辨认的主要对象，按显著程度从高到低排列。
- 同类对象合并为一个名词，不重复列举。

---

## ✅ 语言与格式要求
- 全部使用中文，纯文本输出。
- 禁止输出标题、序号、项目符号、Markdown 标记。
- 禁止输出任何解释性文字、前缀或后缀。
- 禁止换行，两句话连续写在同一段中。
- 行尾不得有空格。

---

## 🚨 兜底规则
若图片完全无法辨识内容（如纯色、严重损坏），
必须只输出：
无法判断。

---

✅ 示例：
室内厨房台面与橱柜区域。画面中包含木质橱柜，不锈钢水槽，灶台，抽油烟机。

✅ 示例：
街边服装摊位前聚集挑选商品的人群。画面中包含短袖套装，人物海报，上衣，短裤，人群。
'''

PROMPT_TEXT_S1 = '''
# 🚨 STAGE 1：对象外观描述与绝对位置（NON-THINKING）

## 🔒 对象生成硬门禁（必须遵守）
在生成任何 1–5 项描述之前，先判断：

是否存在至少一个【连续的、可成形的非纯白像素区域】。

✅ 若存在，继续执行；
❌ 若不存在，必须直接输出：
1、无法判断
2、无法判断
3、无法判断
4、无法判断
5、无法判断
并立即结束，不得生成其他内容。

---

## 🚫 图像作用域隔离（强制）
- 本阶段【只允许】使用 IMAGE_STAGE_1 中的视觉像素。
- IMAGE_STAGE_2 的任何信息一律禁止使用。
- 纯白区域、白色剪影、白色轮廓，一律视为【不存在】，不得描述。

---

## 🧭 类别命名参考（受限上下文）

在图片之外，会提供一段【图片整体描述文本】作为参考：
{json_text}

使用规则（必须遵守）：
- 该文本【仅可用于】第 2 项“类别”的命名选择。
- 仅在【多个类别名称均可合理匹配同一可见像素区域】时，才允许参考该文本。
- 禁止根据该文本生成图片中未能从像素直接看到的对象。
- 禁止根据该文本增加、补全或修改任何外观特征、颜色、结构或细节。
- 禁止根据该文本判断对象数量或对象是否存在。

⚠️ 所有对象是否存在、数量多少、外观如何，
必须始终且仅由 IMAGE_STAGE_1 中的非纯白像素决定。

---

## ✅ 对象与像素基本规则
- 只描述 IMAGE_STAGE_1 中可直接看到的【非纯白像素区域】。
- 灰度、阴影、模糊暗部，视为非纯白像素。
- 不得补全、推断、想象白色区域中的对象。

---

## ✅ 对象数量规则

【连续结构合并规则｜强制】
若多个相似的非纯白像素区域：
- 在空间上首尾相接或紧密相邻，
- 底部或主体部分形成连续整体，
- 材质与类别一致，

则必须合并为【一个对象】描述，
不得因局部形状差异、缺口或顶部变化而拆分为多个对象。

- 若只有一种连续的大面积背景像素区域（如天空、地面、墙面等），只描述【一个背景对象】。
- 若存在多个【明显分离且不满足连续结构合并规则】的非纯白像素区域，
    则描述为【多个对象】。
- 一旦决定对象数量，不得拆分或合并。

---

## ✅ 输出结构（必须严格遵守）
只允许输出 **一组** 1–5 项，不得重复编号。

### 1–5 项含义
1、位置（仅 1 个词：上 / 下 / 左 / 右 / 中间）  
2、类别（具体、可指代的名词，如：轿车、小孩、橱柜、水槽）  
3、短语描述（4–8 字，格式：“…的[类别]”）  
4、详细描述（≤20 字，格式：“…的[第3项原文]”）  
5、绝对位置（格式：“图片[位置]的[第4项原文]”）

---

## ✅ 描述继承硬规则（NON-THINKING 版本）
- 第 4 项必须【完整包含第 3 项的原文文本】。
- 第 5 项必须【完整包含第 4 项的原文文本】。
- 不得删除、替换或改写已出现的词语。

---

## ✅ 多对象并排规则
【对象数量锁定规则｜强制】
一旦在第 1 项或第 2 项中确定存在多个对象，
则第 3、4、5 项中【必须以相同数量并排输出】，
即使多个对象的描述文本完全相同，
也必须重复并排，不得合并为一条。

- 多个对象时，1–5 项 **必须横向并排输出**。
- 使用中文分号 “；” 分隔。
- 并排顺序在所有 5 项中必须完全一致。
- 不得为不同对象重复输出 1–5。

---

## ✅ 语言与格式要求
- 禁止使用主谓宾句式。
- 禁止使用“有 / 是 / 在”等谓语动词。
- 禁止使用中文顿号“、”。
- 每一行结尾不得有空格。
- 输出结果前后不得添加任何解释性文字。

---

## 🌐 英文翻译规则
- 在中文输出后，必须输出英文翻译。
- 英文部分以：
---ENGLISH_TRANSLATION---
开头。
- 英文结构、顺序、对象数量必须与中文完全一致。

---

✅ 单对象示例：
1、中间
2、轿车
3、灰色家用轿车
4、银灰车身带车标的家用轿车
5、图片中间的银灰车身带车标的家用轿车
---ENGLISH_TRANSLATION---
1. center
2. sedan
3. gray family sedan
4. silver-gray sedan with logo on body
5. silver-gray sedan with logo on body in the center of the image

✅ 正确格式示例（两个对象）：
1、上方；下方
2、木质橱柜；不锈钢水槽
3、棕色方形木门橱柜；双槽不锈钢水槽
4、深棕色实木双开门橱柜；银色双槽金属水槽
5、图片上方的深棕色实木双开门橱柜；图片下方的银色双槽金属水槽
---ENGLISH_TRANSLATION---
1. Above; below
2. Wooden cabinets; stainless steel sink
3. Brown square panel cabinets; silver double-basin metal sink
4. Dark brown solid wood double-door cabinets; silver double-basin metal sink
5. The dark brown solid wood double-door cabinets above the image; the silver double-basin metal sink below the image
'''

PROMPT_TEXT_S2 = '''
第一阶段输出为：以上【第一阶段：对象外观描述及其绝对位置】部分的完整输出。
第二阶段中，第一阶段输出的【非方位类描述】（类别、外观、穿着、颜色、配饰、姿态与行为），仅作为【只读对象描述】使用。
第二阶段不得对其进行推理、重组或生成性改写，只能在最终结果中原样复用。
第一阶段输出中出现的所有方位、位置或空间成分
（如左/右/上/下/前/后、中间、边缘、图片××侧等），
在第二阶段中一律视为无效，
不得用于任何空间判断、方向推断或方位组合。

# 🚨 第二阶段：相对位置补充描述（仅在对象描述上锚定第一阶段输出）

【目标定位锚点规则｜最高优先级】
第二阶段中，目标的空间位置判断必须且只能基于 IMAGE_STAGE_2 中
“目标被抹白后的空间占位区域（屏幕坐标）”；
第一阶段中目标在 IMAGE_STAGE_1 中的任何位置、朝向或构图信息，
严禁作为第二阶段左右/上下/前后判断的依据。

【方向词来源强制绑定规则｜执行规则】
第二阶段中所使用的方向词（左 / 右 / 上 / 下 / 前 / 后 / 左前 / 左后 / 右前 / 右后）
必须且只能依据 IMAGE_STAGE_2 中
目标抹白区域与参照物在屏幕坐标上的几何相对关系确定。
严禁基于第一阶段文本、语言习惯或常识推断方向词。

【相对方向单调性约束｜强制执行】
当目标抹白占位区域在 IMAGE_STAGE_2 中
与参照物存在清晰、单向的几何相对关系时：

- 目标整体位于参照物屏幕纵向上方 → 仅允许使用「上方」；
- 目标整体位于参照物屏幕纵向下方 → 仅允许使用「下方」；
- 目标整体位于参照物屏幕横向左侧 → 仅允许使用「左侧」；
- 目标整体位于参照物屏幕横向右侧 → 仅允许使用「右侧」；

❌ 严禁生成与屏幕几何关系相反或不成立的方向词。

【相对方向视角锁定规则｜执行规则】
第二阶段中所有“左 / 右 / 上 / 下 / 前 / 后”等相对方向，
一律表示【目标对象】相对于【参照物】在屏幕坐标中的位置关系。
严禁以参照物、小孩、人物、车或任何非目标对象作为方向判断的视角或原点。

【S1 方向词失效规则｜执行规则】
在第二阶段中，第一阶段文本里出现的所有方向词
（左/右/上/下/前/后、图片左/右/上/下方、left/right/top/bottom 等）
仅用于第一阶段描述，
不表示目标的任何固有属性，
不得参与第二阶段的相对位置判断或方向组合。

【相对位置表达模板强制规则｜执行规则】
第二阶段相对位置描述必须按照如下顺序完成，不得颠倒：
    1）先从 IMAGE_STAGE_2 中确定【方向词】；
    2）再从第一阶段中拷贝【目标对象完整描述】；
    3）最后按以下固定模板拼接输出：
        “参照物 + 的 + 方向词 + 的 + 目标对象”。
禁止在模板拼接阶段重新判断或修改方向词。

【目标唯一性硬约束（Stage2 全局优先级，仅次于目标定位锚点）】
- 当第一阶段第4项**不包含中文分号（；）**时，Stage2 视为**单一目标场景**；
- 在单一目标场景中：
  - **只允许生成一条结果**；
  - ❌ 严禁为同一目标选择多个参照物并输出多条子结果；
  - ❌ 严禁使用分号（； / ;）；
- 若存在多个合理参照物：
  - 仅选择一个**最自然、最近、最显著**的参照物；
  - 其余参照物必须全部舍弃，不得输出。

## 🔑 四原则
1️⃣ **目标零修改+智能拆分**  
   - 单目标：第4项无分隔符 → 整体复用，**禁止拆分或扩展为多个子结果**
   - 多目标：第4项含**中文分号（；）** → 按分隔符拆分子描述（保留子片段原文，不含分隔符）  
   - ❌ 严禁改动任一字词/标点（含子描述内部内容）  
2️⃣ **参照物独立判断（三输入协同+单目标定位）**（每子目标单独执行）
   - 子目标的存在性与语义，严格锚定第一阶段文本输出；
   - 子目标在 IMAGE_STAGE_2 中不可见，其空间位置仅通过“被抹白后的占位区域”确定，
     不得基于任何视觉想象或 Stage1 中的构图印象推断其位置；
   - 参照物**必须**从 IMAGE_STAGE_2（目标被抹白的原图）中选取；
   - 选普通观察者在 IMAGE_STAGE_2 中最自然选用的、最近的显著实体作为参照物；

3️⃣ **方位绝对观察者视角（含前后+左右+上下）**  
   - 严格以**观察者面对屏幕时的视角**为唯一坐标系，与参照物/目标的朝向、运动方向无关：  
     - **左右（强制屏幕坐标优先）**：  
       ✅ **唯一判断标准**：子目标在**屏幕横向左侧区域**→「左」，在**屏幕横向右侧区域**→「右」；  
       【左右判定硬约束】左右仅依据目标与参照物在 IMAGE_STAGE_2 中的屏幕横向相对占位，
        以目标抹白占位区域整体位于参照物左侧或右侧的直观屏幕关系为唯一裁决，
        严禁基于语义、朝向、动作、运动趋势或 Stage1 信息反转左右。
       ❌ **禁止错误逻辑**：禁止参考参照物的“自身左右”（如：车的左边≠屏幕左边）、目标的朝向（如：人朝左时的“左边”≠屏幕左边）；  
       **强制示例**：  
       - 例1：车在图片左侧停放，子目标在车**屏幕右侧**→ 描述为「车右侧的加油泵」（即使车“自身左侧”是屏幕右侧）；  
     - **上下**：屏幕纵向上方=上，屏幕纵向下方=下（无遮挡关系时使用）；  
     - **前后**：仅当目标与参照物存在**遮挡关系**时使用（目标遮挡参照物→「前」；参照物遮挡目标→「后」）；  
   - 中文表述：  
     - 无遮挡：「树上方的目标」「水面下方的船只」；  
     - 有遮挡：「树前方的目标」（目标遮挡树）「水面后方的船只」（水面遮挡船只）；  
   - 英文表述：  
     - 无遮挡：「above the tree」「below the water surface」；  
     - 有遮挡：「in front of the tree」「behind the water surface」；  
   ⚠️【坐标系归属｜最高优先级】  
     - 「前/后/上/下/左/右」是**全局观察者坐标**，**左右方位禁止绑定到参照物/目标的自身朝向**（如：车的左边≠屏幕左边，人的后面≠屏幕后面）；  
     - **视觉示例**：若目标在屏幕上方且遮挡参照物（树）→「树前方的目标」；若目标在屏幕上方但未遮挡树→「树上方的目标」（即使树面朝左）；  
     - **左右方位强制声明**：模型必须**完全忽略参照物/目标的朝向**，仅以屏幕横向坐标判断左右，违反此规则将直接导致输出错误。 

4️⃣ **连接标点硬规范**  
   - 中文行：所有子结果用**中文分号（；）** 连接  
   - 英文行：所有子结果用**英文分号+空格（; ）** 连接  
   - 全局无参照物 → 输出固定短语（非拼接）  

## 📝 单一对象输出（严格仅两行，行尾无空格/换行）
6、[中文结果]  
6. [英文结果]  

## 📝 多对象输出（严格仅两行，行尾无空格/换行）
6、[子结果1]；[子结果2]...  
6. [子结果1英文]; [子结果2英文]...  

## 🌰 多目标示例（第4项="滑水者；船只"）
✅ 正确输出：  
6、树右侧的滑水者；水面左侧的船只  
6. water skier to the right of the tree; boat to the left of the water surface  
❌ 错误：英文用中文分号/中文用英文分号/改动"滑水者"字词  

## ⚠️ 关键执行流
① 拆：检测第4项分隔符 → 生成子描述列表（中英文独立解析）  
② 定（方向冻结）：
- 仅依据 IMAGE_STAGE_2 中
  目标抹白占位区域与参照物在屏幕中的相对位置，
  确定且仅确定一个方向词；
- 一旦方向词确定，
  后续所有步骤【禁止重新判断、比较、修正或校验方向】；
- 后续步骤中若发现潜在冲突，
  以②中确定的方向词为最终裁决。

③ 拼：  
   - 中文：`[参照物][方位]的[子描述]` → 分号连接  
   - 英文：`[子描述英文] [direction] the [reference_EN]` → 英文分号+空格连接；
  [direction] 为不可拆分的整体方向短语（left of / right of / in front of / behind），
  修正方位时必须整体替换。
 
④ 查（必做）：
  - 以观察者面对图片时的屏幕坐标（左/右/上/下/前/后）为唯一依据；
  - 仅比较目标与参照物在屏幕上的相对位置或遮挡关系；
  - 若方位词不一致，仅修正方位词，其余内容保持不变；
  - ❌ 不得更换参照物、子描述或句式；

【兜底输出规则｜必须遵守】
若 IMAGE_STAGE_2 中无可靠参照物，
但抹白占位区域在画面中具有明确方向（左/右/上/下），
仍需基于其相对于画面边缘的位置输出方位描述，
不得参考 IMAGE_STAGE_1，不得无输出。

！！！ 严禁添加解释/换行/修改子描述/混用标点 ！！！  
✅ 口诀：拆得准→盯得紧→连得对（中分号、英分号）
'''


def build_stage1_stage2_prompt(stage1_prompt: str) -> str:
    """
    将 Stage‑1 Prompt + Stage‑2 Prompt 合并成一个完整 Prompt
    """
    return f'''
🚫【全局阶段与视觉隔离总规则｜最高优先级】
本任务包含两个独立的视觉阶段：第一阶段（IMAGE_STAGE_1）与第二阶段（IMAGE_STAGE_2），
两阶段在视觉证据与语义推断上实行**强隔离**。

  - 第一阶段生成内容时，必须且只能使用 IMAGE_STAGE_1 作为唯一视觉依据；
    严禁参考、分析、推断、提及或暗示任何与 IMAGE_STAGE_2 相关的信息。
  - 第二阶段生成内容时，必须严格遵守其专属视觉使用规则，不得反向借用第一阶段的空间或构图信息。
  - 任何阶段仅可使用其被明确授权的图像作为视觉依据，违规视为错误输出。


# 🎯 任务总览
本任务包含两个阶段：  
- Stage 1：基于`IMAGE_STAGE_1`生成对象描述；  
- Stage 2：基于两张图像生成相对位置描述：
  - IMAGE_STAGE_1：仅用于确认目标对象；
  - IMAGE_STAGE_2：目标对象被抹白的原始完整图像，仅用于寻找参照物与判断相对位置；
**Stage 2 核心规则**：  
1. `IMAGE_STAGE_2`中被抹白的区域仅作为目标的空间占位锚点；
   第二阶段中所有左右/上下/前后判断，必须且只能基于该抹白占位区域在屏幕中的位置关系进行。  
2. 方位判断严格基于子目标的屏幕坐标，忽略参照物朝向；  

⚠️ 第一阶段与第二阶段之间的视觉证据使用规则：
- 不允许跨阶段使用“未授权”的视觉证据；
- 第一阶段严禁使用 IMAGE_STAGE_2 的任何视觉信息；
- 第一阶段严禁描述任何非 IMAGE_STAGE_1 中实际可见的对象或区域，
  包括但不限于白色剪影、轮廓、阴影、空白占位等无彩色像素支撑的内容；
- 第一阶段严禁因另一阶段图像更清晰而进行联想、补全或修正判断；

- 第二阶段允许使用 IMAGE_STAGE_1，但仅限于确认目标对象的语义身份（名词本体），
  IMAGE_STAGE_1 中的任何空间、构图或方位信息在第二阶段中视为不可用，包括第一阶段文本中出现的所有方向词；
- 不允许跨阶段借用、对照或补全对象的外观或位置。

任何阶段只能使用其明确指定的图像作为唯一视觉依据。
违反该规则的描述视为错误输出。

【IMAGE_STAGE_1】
- 仅包含目标对象，背景为纯白
- 第一阶段：用于对象外观与绝对位置描述；
- 第二阶段：仅用于目标语义确认，不提供任何空间或方位信息；

【IMAGE_STAGE_2】
- 原始完整图像，目标对象已被抹白
- 仅用于第二阶段寻找参照物与相对位置判断
- 禁止在该图中识别或推断目标对象本身

====================================================
🔹 第一阶段：对象外观描述及其绝对位置（只看 IMAGE_STAGE_1）
====================================================

**【第一阶段内容限定｜强制】**
- 描述的所有对象、特征、颜色、位置等信息，**必须百分之百**来源于 `IMAGE_STAGE_1` 的实际可见像素。
- **严禁**描述任何在 `IMAGE_STAGE_1` 中未出现，但可能存在于 `IMAGE_STAGE_2` 中的物体或概念。
- **禁止**进行任何跨阶段的联想、推测或补全。
- **禁止**提及任何在 `IMAGE_STAGE_1` 中不存在的“轮廓”、“剪影”、“阴影”等无彩色像素支撑的区域。

{stage1_prompt}

====================================================
🔹 第二阶段：相对位置补充（严格锚定第一阶段输出）
====================================================
【第二阶段三输入说明｜必读】

第二阶段同时接收以下三项信息：
- 第一阶段的完整文本输出
- IMAGE_STAGE_1（仅目标为彩色，其余为纯白）：仅用于目标语义确认，不提供任何空间或方位线索
- IMAGE_STAGE_2（目标对象被抹白的原图）：仅用于寻找参照物与判断相对方位

⚠️ 目标对象在 IMAGE_STAGE_2 中不可见，禁止在其中寻找目标。

{PROMPT_TEXT_S2}
'''


def get_context_from_json(json_path: str) -> str:
    """
    从 json 文件中读取 global_caption 的第一句话
    若文件或字段不存在，返回空字符串
    """
    if not os.path.exists(json_path):
        return ""

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        caption = data.get("global_caption", "")
        if not caption:
            return ""

        return get_first_sentence(caption)

    except Exception:
        return ""


def get_first_sentence(caption: str) -> str:
    """
    取描述文本的第一句话（按中文句号切分）
    """
    if not caption:
        return ""

    # 统一去除首尾空白
    caption = caption.strip()

    # 优先按中文句号切分
    if "。" in caption:
        return caption.split("。", 1)[0] + "。"

    # 没有句号则直接返回
    return caption


def get_image_base654(image_path):
    file_path = Path(image_path)
    file_extension = file_path.suffix
    with open(image_path, "rb") as image_file:
        base64_str = base64.b64encode(image_file.read()).decode('utf-8')
    return file_extension[1:], base64_str


def resize_keep_ratio_long_side(image, long_side=IMAGE_LONG_SIDE):
    """
    保持宽高比，将图片长边 resize 到 long_side
    长边已不大于 long_side 时不做放大，避免无意义的 token 消耗
    """
    h, w = image.shape[0:2]
    factor = long_side / max(h, w)
    if factor >= 1.0:
        return image

    resize_h, resize_w = int(round(h * factor)), int(round(w * factor))

    return cv2.resize(image, (resize_w, resize_h),
                      interpolation=cv2.INTER_AREA)


def encode_image_to_base64(image):
    """
    将内存中的 BGR 图片编码为 jpeg base64 字符串（不落盘）
    """
    encode_flag, buffer = cv2.imencode(
        '.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), IMAGE_JPEG_QUALITY])
    if not encode_flag:
        return None

    return base64.b64encode(buffer.tobytes()).decode('utf-8')


def decode_annotation_mask(annotation, image_h, image_w):
    """
    将 COCO RLE 格式的 segmentation 解码为 bool mask
    尺寸与图片不一致时返回 None
    """
    per_mask = mask_utils.decode(annotation['segmentation'])
    if per_mask is None:
        return None

    if per_mask.ndim == 3:
        per_mask = per_mask[:, :, 0]

    if per_mask.shape[0] != image_h or per_mask.shape[1] != image_w:
        return None

    return per_mask.astype(bool)


def build_stage_images(image, mask):
    """
    由原图与目标 mask 现场合成两张 stage 图（内存中完成，不落盘）
        IMAGE_STAGE_1: 仅保留目标像素，其余区域涂成纯白
        IMAGE_STAGE_2: 目标区域抹白，其余保留原图
    """
    stage1_image = image.copy()
    stage1_image[~mask] = 255

    stage2_image = image.copy()
    stage2_image[mask] = 255

    return stage1_image, stage2_image


def build_image_content(base64_str):
    """
    构造 DeepSeek API 的图片内容块
    """
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_str}",
            "detail": IMAGE_DETAIL
        }
    }


def request_deepseek_api(content_list):
    """
    调用 DeepSeek chat completions 接口（deepseek-flash 支持图像理解）
    对 429 与 5xx 做指数退避重试
    返回: {"ok": bool, "content": str} 或 {"ok": False, "error": str}
    """
    body = {
        "model": DEEPSEEK_MODEL,
        "stream": False,
        "messages": [{
            "role": "user",
            "content": content_list
        }],
        "thinking": {
            "type": "disabled"
        },
        "max_tokens": MAX_OUTPUT_TOKENS
    }
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}'
    }

    last_error = "unknown_error"
    for per_retry_idx in range(MAX_RETRY_TIMES):
        try:
            response = requests.post(
                DEEPSEEK_URL,
                json=body,
                headers=headers,
                timeout=REQUEST_TIMEOUT  # ✅ must have timeout
            )

            if response.status_code != 200:
                last_error = f"http_{response.status_code}"
                # ✅ 429 为并发超限, 5xx 为服务端问题, 均可重试
                if response.status_code == 429 or response.status_code >= 500:
                    if per_retry_idx < MAX_RETRY_TIMES - 1:
                        time.sleep(RETRY_BACKOFF_SECONDS * (2**per_retry_idx) +
                                   random.uniform(0, 0.5))
                        continue
                return {"ok": False, "error": last_error}

            try:
                data = response.json()
            except Exception:
                return {"ok": False, "error": "invalid_json"}

            try:
                content = data["choices"][0]["message"]["content"]
            except Exception:
                return {"ok": False, "error": "missing_content"}

            if not content or not isinstance(content, str):
                return {"ok": False, "error": "empty_content"}

            return {"ok": True, "content": content}

        except Exception as e:
            last_error = type(e).__name__
            if per_retry_idx < MAX_RETRY_TIMES - 1:
                time.sleep(RETRY_BACKOFF_SECONDS * (2**per_retry_idx) +
                           random.uniform(0, 0.5))
                continue

    return {"ok": False, "error": last_error}


def inference(base64_str_s1, base64_str_s2, prompt):
    """
    两阶段描述推理
    ⚠️ 文本 prompt 必须放在图片之前, 使公共 prompt 前缀能命中 KVCache, 大幅降低费用
    """
    tmp_content = []

    # ====== PROMPT ======
    tmp_content.append({"type": "text", "text": prompt})

    # ===== IMAGE_STAGE_1 =====
    tmp_content.append(build_image_content(base64_str_s1))

    # ===== IMAGE_STAGE_2 =====
    tmp_content.append(build_image_content(base64_str_s2))

    return request_deepseek_api(tmp_content)


def inference_global_caption(base64_str_image, prompt):
    """
    整张图像的全局描述推理（单图输入）
    """
    tmp_content = [
        {
            "type": "text",
            "text": prompt
        },
        build_image_content(base64_str_image),
    ]

    return request_deepseek_api(tmp_content)


def validate_output_format(output_text: str) -> dict:
    """
    验证输出格式是否符合要求
    返回: {"valid": bool, "error": str or None}
    """
    # ✅ 检查是否包含英文翻译分隔符
    if "---ENGLISH_TRANSLATION---" not in output_text:
        return {
            "valid": False,
            "error": "missing_english_translation_separator"
        }

    cn_part, en_part = output_text.split("---ENGLISH_TRANSLATION---", 1)
    cn_lines = [l for l in cn_part.strip().splitlines() if l.strip()]
    en_lines = [l for l in en_part.strip().splitlines() if l.strip()]

    # ✅ 检查中文部分：必须包含1-6点
    cn_numbers = []
    for line in cn_lines:
        match = re.match(r"^([1-6])、", line)
        if match:
            cn_numbers.append(match.group(1))

    if set(cn_numbers) != {'1', '2', '3', '4', '5', '6'}:
        missing = set(['1', '2', '3', '4', '5', '6']) - set(cn_numbers)
        return {
            "valid": False,
            "error": f"missing_chinese_points_{','.join(sorted(missing))}"
        }

    # ✅ 检查英文部分：必须包含1-6点
    en_numbers = []
    for line in en_lines:
        match = re.match(r"^([1-6])\.", line)
        if match:
            en_numbers.append(match.group(1))

    if set(en_numbers) != {'1', '2', '3', '4', '5', '6'}:
        missing = set(['1', '2', '3', '4', '5', '6']) - set(en_numbers)
        return {
            "valid": False,
            "error": f"missing_english_points_{','.join(sorted(missing))}"
        }

    return {"valid": True, "error": None}


def reorder_stage2_chinese(output_text: str) -> str:
    """
    将二阶段中文第6点（若出现在英文区块中），
    移动到中文第5点之后
    """
    if "---ENGLISH_TRANSLATION---" not in output_text:
        return output_text

    cn_part, en_part = output_text.split("---ENGLISH_TRANSLATION---", 1)

    cn_lines = [l for l in cn_part.strip().splitlines() if l.strip()]
    en_lines = [l for l in en_part.strip().splitlines() if l.strip()]

    # ✅ 从英文区块中找中文第6点
    stage2_cn = None
    for line in en_lines:
        if re.match(r"^6、", line):
            stage2_cn = line
            break

    # 没找到就直接返回
    if not stage2_cn:
        return output_text

    # ✅ 从英文区块中移除该中文行
    en_lines = [l for l in en_lines if l != stage2_cn]

    # ✅ 插入到中文第5点之后
    insert_idx = 5 if len(cn_lines) >= 5 else len(cn_lines)
    cn_lines.insert(insert_idx, stage2_cn)

    # ✅ 重新拼接
    new_cn = "\n".join(cn_lines)
    new_en = "\n".join(en_lines)

    return f"{new_cn}\n---ENGLISH_TRANSLATION---\n{new_en}"


def process_one_global_caption(args):
    """
    单张图片的全局描述处理流程
    ✅ 同一张图片的全局描述只获取一次, 已存在且 status == ok 时直接跳过
    """
    image_name, image_path, save_path = args

    try:
        # ✅ 已获取过则直接复用, 不重复调用 API
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r', encoding='utf-8') as f:
                    exist_data = json.load(f)
                if exist_data.get('status') == 'ok' and exist_data.get(
                        'global_caption'):
                    return {
                        "image_name": image_name,
                        "status": "skip",
                    }
            except Exception:
                pass

        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None:
            return {
                "image_name": image_name,
                "image_path": image_path,
                "status": "error",
                "error": "imread_failed"
            }

        # ✅ 原图保持宽高比, 长边 resize 到 1024
        base64_str_image = encode_image_to_base64(
            resize_keep_ratio_long_side(image))
        if base64_str_image is None:
            return {
                "image_name": image_name,
                "image_path": image_path,
                "status": "error",
                "error": "imencode_failed"
            }

        infer_ret = inference_global_caption(base64_str_image,
                                             PROMPT_TEXT_GLOBAL)

        if not infer_ret.get("ok"):
            return {
                "image_name": image_name,
                "image_path": image_path,
                "status": "error",
                "error": infer_ret.get("error", "unknown_inference_error")
            }

        global_caption = infer_ret["content"].strip()

        single_output = {
            "image_name": image_name,
            "image_path": image_path,
            "global_caption": global_caption,
            "status": "ok"
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(single_output, f, ensure_ascii=False, indent=2)

        return {
            "image_name": image_name,
            "image_path": image_path,
            "global_caption": global_caption,
            "status": "ok"
        }

    except Exception as e:
        # ✅ last-resort catch (still no raise)
        return {
            "image_name": image_name,
            "image_path": image_path,
            "status": "error",
            "error": type(e).__name__
        }


def process_one_item(args):
    """
    单个样本的完整处理流程（用于多进程）
    两张 stage 图由原图 + RLE mask 在内存中现场合成, 不落盘
    """
    (sample_id, image_path, json_path, ann_idx, json_text, save_path) = args

    try:
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8),
                             cv2.IMREAD_COLOR)
        if image is None:
            return {
                "sample_id": sample_id,
                "image_path": image_path,
                "json_path": json_path,
                "ann_idx": ann_idx,
                "context": json_text,
                "status": "error",
                "error": "imread_failed"
            }

        with open(json_path, 'r', encoding='utf-8') as f:
            anno_data = json.load(f)

        annotation = anno_data['annotations'][ann_idx]
        mask = decode_annotation_mask(annotation, image.shape[0],
                                      image.shape[1])
        if mask is None or not mask.any():
            return {
                "sample_id": sample_id,
                "image_path": image_path,
                "json_path": json_path,
                "ann_idx": ann_idx,
                "context": json_text,
                "status": "error",
                "error": "invalid_mask"
            }

        # ✅ 内存中合成两张 stage 图, 并保持宽高比 resize 到长边 1024
        stage1_image, stage2_image = build_stage_images(image, mask)
        base64_str_s1 = encode_image_to_base64(
            resize_keep_ratio_long_side(stage1_image))
        base64_str_s2 = encode_image_to_base64(
            resize_keep_ratio_long_side(stage2_image))
        if base64_str_s1 is None or base64_str_s2 is None:
            return {
                "sample_id": sample_id,
                "image_path": image_path,
                "json_path": json_path,
                "ann_idx": ann_idx,
                "context": json_text,
                "status": "error",
                "error": "imencode_failed"
            }

        stage1_prompt = PROMPT_TEXT_S1.format(
            json_text=json_text if json_text else "（无可用上下文描述）")
        full_prompt = build_stage1_stage2_prompt(stage1_prompt)

        infer_ret = inference(base64_str_s1, base64_str_s2, full_prompt)

        # ✅ inference-level failure
        if not infer_ret.get("ok"):
            return {
                "sample_id": sample_id,
                "image_path": image_path,
                "json_path": json_path,
                "ann_idx": ann_idx,
                "context": json_text,
                "status": "error",
                "error": infer_ret.get("error", "unknown_inference_error")
            }

        r = infer_ret["content"]
        r = reorder_stage2_chinese(r)

        # ✅ 格式检查
        format_check = validate_output_format(r)
        if not format_check["valid"]:
            return {
                "sample_id": sample_id,
                "image_path": image_path,
                "json_path": json_path,
                "ann_idx": ann_idx,
                "context": json_text,
                "output_text": r,
                "status": "format_error",
                "error": format_check["error"]
            }

        single_output = {
            "sample_id": sample_id,
            "image_path": image_path,
            "json_path": json_path,
            "ann_idx": ann_idx,
            "context": json_text,
            "output_text": r,
            "status": "ok"
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(single_output, f, ensure_ascii=False, indent=2)

        return {
            "sample_id": sample_id,
            "image_path": image_path,
            "json_path": json_path,
            "ann_idx": ann_idx,
            "context": json_text,
            "prompt": r,
            "status": "ok"
        }

    except Exception as e:
        # ✅ last-resort catch (still no raise)
        return {
            "sample_id": sample_id,
            "image_path": image_path,
            "json_path": json_path,
            "ann_idx": ann_idx,
            "context": json_text,
            "status": "error",
            "error": type(e).__name__
        }


def process_item_batch(args_batch):
    """
    进程内用线程池并发发起 API 请求
    CPU 预处理在进程间并行, 网络 IO 在进程内线程间并发
    """
    if not args_batch:
        return []

    thread_num = min(THREAD_PER_PROC, len(args_batch))
    with ThreadPool(thread_num) as thread_pool:
        return list(thread_pool.imap(process_one_item, args_batch))


def process_global_caption_batch(args_batch):
    """
    进程内用线程池并发获取全局描述
    """
    if not args_batch:
        return []

    thread_num = min(THREAD_PER_PROC, len(args_batch))
    with ThreadPool(thread_num) as thread_pool:
        return list(thread_pool.imap(process_one_global_caption, args_batch))


def split_to_batches(item_list, batch_size):
    """
    将任务列表切分为若干批次
    """
    return [
        item_list[i:i + batch_size]
        for i in range(0, len(item_list), batch_size)
    ]


def get_subset_split_dir(subset_name):
    """
    数据集中某个子集的数据目录
    """
    return os.path.join(DATASET_ROOT, subset_name, SPLIT_NAME)


def get_subset_save_dir(subset_name):
    """
    某个子集的结果保存目录
    """
    return os.path.join(CAPTION_SAVE_ROOT, subset_name, SPLIT_NAME)


def collect_image_name_list(subset_name):
    """
    收集某个子集下所有有效的图片名（jpg 与 json 必须同时存在）
    """
    split_dir = get_subset_split_dir(subset_name)
    if not os.path.exists(split_dir):
        print(f'❌ subset dir not found: {split_dir}')
        return []

    with os.scandir(split_dir) as entries:
        all_json_name_list = sorted(
            [e.name for e in entries if e.name.endswith('.json')])

    image_name_list = []
    for per_json_name in all_json_name_list:
        per_image_name = per_json_name[:-5]
        if not os.path.exists(os.path.join(split_dir,
                                           per_image_name + '.jpg')):
            continue
        image_name_list.append(per_image_name)

    return image_name_list


def build_global_captions(pool_num, subset_to_image_name_dict):
    """
    ✅ 第一步: 单独把原图 resize 到长边 1024, 与文字 prompt 一起输入获取全局描述
    同一张图片只获取一次, 结果落盘后供该图片的所有 mask 复用
    返回: {(subset_name, image_name): global_caption}
    """
    caption_task_list = []
    for per_subset_name, per_image_name_list in subset_to_image_name_dict.items(
    ):
        per_split_dir = get_subset_split_dir(per_subset_name)
        per_save_dir = get_subset_save_dir(per_subset_name)
        os.makedirs(per_save_dir, exist_ok=True)

        for per_image_name in per_image_name_list:
            caption_task_list.append((
                per_image_name,
                os.path.join(per_split_dir, per_image_name + '.jpg'),
                os.path.join(per_save_dir,
                             per_image_name + GLOBAL_CAPTION_SUFFIX),
            ))

    print(f'Total global caption tasks: {len(caption_task_list)}')

    ok_count, skip_count, error_count = 0, 0, 0
    if caption_task_list:
        batch_list = split_to_batches(caption_task_list, THREAD_PER_PROC)
        with Pool(pool_num) as pool:
            for per_batch_ret in tqdm(pool.imap(process_global_caption_batch,
                                                batch_list),
                                      total=len(batch_list),
                                      desc='Getting global captions'):
                for per_ret in per_batch_ret:
                    if per_ret.get('status') == 'ok':
                        ok_count += 1
                    elif per_ret.get('status') == 'skip':
                        skip_count += 1
                    else:
                        error_count += 1

    print(f'Global captions -> new: {ok_count}, reused: {skip_count}, '
          f'failed: {error_count}')

    # ✅ 统一载入内存, 供后续 mask 级任务复用
    global_caption_dict = {}
    for per_subset_name, per_image_name_list in subset_to_image_name_dict.items(
    ):
        per_save_dir = get_subset_save_dir(per_subset_name)
        for per_image_name in per_image_name_list:
            per_caption_path = os.path.join(
                per_save_dir, per_image_name + GLOBAL_CAPTION_SUFFIX)
            global_caption_dict[(
                per_subset_name,
                per_image_name)] = get_context_from_json(per_caption_path)

    return global_caption_dict


def collect_mask_pair_tasks(subset_to_image_name_dict, global_caption_dict):
    """
    ✅ 第二步: 收集所有图片-mask 对任务
    按 mask 面积比例过滤, 并跳过已完成的样本（支持断点续跑）
    """
    image_pairs = []
    for per_subset_name, per_image_name_list in subset_to_image_name_dict.items(
    ):
        per_split_dir = get_subset_split_dir(per_subset_name)
        per_save_dir = get_subset_save_dir(per_subset_name)

        for per_image_name in tqdm(
                per_image_name_list,
                desc=f'Collecting data [{per_subset_name}]'):
            per_image_path = os.path.join(per_split_dir,
                                          per_image_name + '.jpg')
            per_json_path = os.path.join(per_split_dir,
                                         per_image_name + '.json')

            # ✅ now safe to read JSON
            try:
                with open(per_json_path, 'r', encoding='utf-8') as f:
                    per_anno_data = json.load(f)
                per_image_h = per_anno_data['image']['height']
                per_image_w = per_anno_data['image']['width']
                per_annotation_list = per_anno_data['annotations']
            except Exception:
                # ✅ bad json → skip sample
                continue

            per_image_area = per_image_h * per_image_w
            if per_image_area <= 0:
                continue

            per_json_text = global_caption_dict.get(
                (per_subset_name, per_image_name), '')

            for per_ann_idx, per_annotation in enumerate(per_annotation_list):
                # ✅ 过滤面积过小的碎片 mask
                per_area_ratio = per_annotation.get('area', 0) / per_image_area
                if per_area_ratio <= MIN_MASK_AREA_RATIO:
                    continue

                per_save_path = os.path.join(
                    per_save_dir,
                    f'{per_image_name}_{per_ann_idx}{OUTPUT_JSON_SUFFIX}')

                # ✅ 已完成的样本直接跳过, 支持断点续跑
                if os.path.exists(per_save_path):
                    try:
                        with open(per_save_path, 'r', encoding='utf-8') as f:
                            per_exist_data = json.load(f)
                        if per_exist_data.get('status') == 'ok':
                            continue
                    except Exception:
                        pass

                per_sample_id = f'{per_subset_name}_{per_image_name}_{per_ann_idx}'
                image_pairs.append(
                    (per_sample_id, per_image_path, per_json_path, per_ann_idx,
                     per_json_text, per_save_path))

    return image_pairs


def main(pool_num=2, rerun_false_cases=False, error_log_file=None):

    # ✅ 创建日志目录
    log_dir = LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    # ✅ 生成错误日志文件路径
    error_log_path = os.path.join(log_dir, f'error_log_{TIMESTAMP}.json')

    # ✅ 生成统计日志文件路径
    stats_log_path = os.path.join(log_dir, f'stats_log_{TIMESTAMP}.json')

    # ✅ 初始化错误日志文件（写入空数组）
    with open(error_log_path, 'w', encoding='utf-8') as f:
        json.dump([], f, ensure_ascii=False, indent=2)

    # ✅ 1. 收集所有任务（单线程）
    image_pairs = []

    # ✅ 重跑错误数据模式
    if rerun_false_cases:
        if not error_log_file or not os.path.exists(error_log_file):
            print(f"❌ Error log file not found: {error_log_file}")
            return

        print(f"🔄 Rerunning failed cases from: {error_log_file}")

        try:
            with open(error_log_file, 'r', encoding='utf-8') as f:
                error_data = json.load(f)
        except Exception as e:
            print(f"❌ Failed to read error log: {e}")
            return

        print(f"📊 Found {len(error_data)} failed cases")

        for item in tqdm(error_data, desc="Loading failed cases"):
            image_path = item.get("image_path")
            json_path = item.get("json_path")
            ann_idx = item.get("ann_idx")
            json_text = item.get("context", "")

            if not image_path or not json_path or ann_idx is None:
                continue

            if not os.path.exists(image_path) or not os.path.exists(json_path):
                continue

            sample_id = item.get("sample_id", str(len(image_pairs)))
            image_name = os.path.basename(image_path)[:-4]
            save_path = os.path.join(
                os.path.dirname(
                    image_path.replace(DATASET_ROOT, CAPTION_SAVE_ROOT)),
                f'{image_name}_{ann_idx}{OUTPUT_JSON_SUFFIX}')

            image_pairs.append((sample_id, image_path, json_path, ann_idx,
                                json_text, save_path))

    # ✅ 正常模式：遍历数据集子集
    else:
        subset_to_image_name_dict = {}
        for per_subset_name in SUBSET_NAME_LIST:
            subset_to_image_name_dict[per_subset_name] = \
                collect_image_name_list(per_subset_name)
            print(f'subset [{per_subset_name}]: '
                  f'{len(subset_to_image_name_dict[per_subset_name])} images')

        # ✅ 第一步: 获取整张图像的全局描述（每张图只获取一次, 后续复用）
        global_caption_dict = build_global_captions(pool_num,
                                                    subset_to_image_name_dict)

        # ✅ 第二步: 收集所有图片-mask 对任务
        image_pairs = collect_mask_pair_tasks(subset_to_image_name_dict,
                                              global_caption_dict)

    print(f"Total tasks: {len(image_pairs)}")

    total_items = len(image_pairs)

    # ✅ 2. 多进程 + 进程内多线程调用 API（保持顺序）
    # ✅ 统计变量
    success_count = 0
    error_count = 0
    format_error_count = 0
    error_logs = []

    batch_list = split_to_batches(image_pairs, THREAD_PER_PROC)
    pool = Pool(pool_num)

    processed_count = 0
    pbar = tqdm(total=total_items, desc="Calling API")
    for per_batch_ret in pool.imap(process_item_batch,
                                   batch_list):  # ✅ 用 imap 保持顺序
        for r in per_batch_ret:
            if r.get("status") == "ok":
                success_count += 1
            elif r.get("status") == "format_error":
                format_error_count += 1
                error_logs.append(r)
            else:
                error_count += 1
                error_logs.append(r)

        processed_count += len(per_batch_ret)
        pbar.update(len(per_batch_ret))

        # ✅ 实时保存错误日志（按批次写入, 避免逐条写盘拖慢高并发）
        if error_logs:
            with open(error_log_path, 'w', encoding='utf-8') as f:
                json.dump(error_logs, f, ensure_ascii=False, indent=2)

    pbar.close()
    pool.close()
    pool.join()

    # ✅ 3. 保存统计结果到JSON文件
    stats = {
        "timestamp":
        TIMESTAMP,
        "data_dir":
        f'{DATASET_ROOT} {SUBSET_NAME_LIST}'
        if not rerun_false_cases else f"RERUN from {error_log_file}",
        "save_dir":
        CAPTION_SAVE_ROOT,
        "model":
        DEEPSEEK_MODEL,
        "image_long_side":
        IMAGE_LONG_SIDE,
        "min_mask_area_ratio":
        MIN_MASK_AREA_RATIO,
        "pool_num":
        pool_num,
        "thread_per_proc":
        THREAD_PER_PROC,
        "total_tasks":
        total_items,
        "success_count":
        success_count,
        "error_count":
        error_count,
        "format_error_count":
        format_error_count,
        "success_rate":
        f"{success_count / total_items * 100:.2f}%"
        if total_items > 0 else "0%",
        "error_log_path":
        error_log_path,
        "rerun_mode":
        rerun_false_cases
    }

    with open(stats_log_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    # ✅ 4. 输出统计信息到控制台
    if error_count > 0 or format_error_count > 0:
        print(f"\n❌ Errors: {error_count} samples failed")
        print(f"⚠️  Format Errors: {format_error_count} samples")
        print(f"Error log saved to: {error_log_path}")

    print(f"\n✅ Success: {success_count} samples processed")
    print(f"Stats log saved to: {stats_log_path}")
    print(f"Processing completed. Results saved in {CAPTION_SAVE_ROOT}.")


if __name__ == "__main__":
    RERUN_FALSE_CASES = False
    ERROR_LOG_FILE = None

    main(pool_num=POOL_NUM,
         rerun_false_cases=RERUN_FALSE_CASES,
         error_log_file=ERROR_LOG_FILE)
