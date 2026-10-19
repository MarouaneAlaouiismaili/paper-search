from dataclasses import dataclass

@dataclass
class Config:
    request_delay: float = 1.0

# كائن إعدادات عام (Global Config) للوصول له من دوال الشبكة
global_config = Config()
