"""国产算力适配层的统一推理接口。

架构断言
--------
    换芯片 = 换执行后端（Execution Provider）

业务代码（``io`` / ``tasks`` / ``engine`` / ``output`` / ``viz``）只依赖本模块的
:class:`InferenceBackend`，**不得直接 import onnxruntime 或任何厂商 SDK**。
跨芯片适配的全部工作量，就是在 ``runtime/`` 目录下新增一个后端实现。

设计约束
--------
- 后端之间只交换 numpy 数组，不传递 torch / 厂商张量对象。
- 后端实例一律通过 :func:`registry.create` 创建，业务代码不直接实例化具体类。
- ``describe()`` 的返回值直接汇入《国产化适配测试报告》的"环境清单表"。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np


class BackendError(RuntimeError):
    """后端加载或推理失败。"""


class InferenceBackend(ABC):
    """推理后端统一接口。

    子类须实现 :meth:`load` / :meth:`infer` / :meth:`describe`，
    并在类上声明唯一的类属性 ``name``。
    """

    #: 后端标识，用于配置与注册表查找，例如 "ort" / "rknn" / "cann"
    name: str = "base"

    def __init__(self, **options: Any) -> None:
        self.options = options
        self._loaded = False

    @abstractmethod
    def load(self, model_path: str, **kwargs: Any) -> None:
        """加载模型并完成初始化。

        注意：真正的初始化开销（运行时创建、kernel 选择、显存池分配）往往发生在
        **第一次推理**而非此处。做时延基准前必须先调用 :meth:`warmup`，
        否则一次性开销会被计入单次时延（本项目已因此踩坑，见 docs/decisions.md）。
        """

    @abstractmethod
    def infer(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """执行一次推理。

        :param inputs: 键为模型输入名，值为 numpy 数组
        :return: 键为模型输出名，值为 numpy 数组
        """

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """返回本后端的环境描述。

        必须包含：后端标识、运行时及版本、实际生效的执行后端/设备、
        以及**是否包含 NVIDIA 组件**（国产化自证的关键字段）。
        """

    def warmup(self, inputs: Mapping[str, np.ndarray], times: int = 5) -> None:
        """用真实输入预热，烧掉一次性开销。"""
        for _ in range(times):
            self.infer(inputs)

    @property
    def ready(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} ready={self._loaded}>"
