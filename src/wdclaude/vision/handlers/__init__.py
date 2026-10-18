"""媒体 Handler 实现包（handler_class 的动态加载目标包，spec 8.1 / 9）。

本包只承载各媒体类型的 Handler 实现类；kind -> Handler 的映射由
config.yaml 的 media.handlers 段 + MediaRegistry 装配，本包不维护映射。

未来新增媒体类型（Phase 4）：在本包内实现对应 Handler 类
（如 video_handler.VideoHandler），并在 media.handlers 中增加该段配置
（enabled 改 true 并解除注释），框架代码零改动。
"""

from wdclaude.vision.handlers.image_handler import ImageHandler

__all__ = ["ImageHandler"]
