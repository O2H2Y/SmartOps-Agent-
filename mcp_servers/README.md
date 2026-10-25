# MCP Servers

为 AIOps 智能诊断提供日志查询和监控数据工具。

## 📚 服务列表

### CLS Server (`cls_server.py`)
**日志查询服务** - 端口 8003

**核心工具：**
- `get_current_timestamp` - 获取当前时间戳
- `get_region_code_by_name` - 查询地区代码
- `get_topic_info_by_name` - 按名称查询日志主题
- `search_topic_by_service_name` - 按服务名查询日志主题
- `search_log` - 按主题和时间范围搜索日志

### Monitor Server (`monitor_server.py`)
**监控数据服务** - 端口 8004

**核心工具：**
- `query_cpu_metrics` - CPU 使用率查询
- `query_memory_metrics` - 内存使用查询

## 🚀 快速开始

### 安装依赖
```bash
pip install fastmcp
```

### 启动服务

**方式一：使用 Makefile（推荐）**
```bash
make start-cls       # 启动 CLS MCP 服务
make start-monitor   # 启动 Monitor MCP 服务
make status-mcp      # 查看 MCP 服务状态
```

**方式二：手动启动**
```bash
python mcp_servers/cls_server.py
python mcp_servers/monitor_server.py
```

## 💡 使用示例

### AIOps 诊断场景

```
用户: data-sync-service 出现告警，请排查

Agent 自动执行:
1. search_topic_by_service_name("data-sync-service") → 查找日志主题
2. get_current_timestamp() → 获取当前毫秒时间戳
3. search_log(topic_id, start_time, end_time) → 查询近期日志
4. query_cpu_metrics("data-sync-service") → CPU 趋势分析
5. query_memory_metrics("data-sync-service") → 内存趋势分析
6. 综合分析 → 生成诊断报告和修复建议
```

### 工具参数示例

**查询 CPU 指标：**
```python
query_cpu_metrics(
    service_name="data-sync-service",
    start_time="2024-02-14 02:00:00",
    interval="1m"
)
```

**搜索错误日志：**
```python
current = get_current_timestamp()
start = current - 15 * 60 * 1000
search_log(
    topic_id="topic-001",
    start_time=start,
    end_time=current,
    query="level:ERROR",
    limit=100,
)
```

## 🔧 高级配置

### 接入真实 API

当前返回模拟数据。接入真实 API 步骤：

**腾讯云 CLS：**
```bash
# 安装 SDK
pip install tencentcloud-sdk-python

# 配置环境变量
export TENCENTCLOUD_SECRET_ID="your-id"
export TENCENTCLOUD_SECRET_KEY="your-key"

# 在 cls_server.py 中集成
from tencentcloud.cls.v20201016 import cls_client
```

**其他监控系统：**
- Prometheus
- Grafana
- 云监控（腾讯云/阿里云/AWS）
- 自建监控平台

### 自定义 Mock 数据

修改各 Server 文件中的数据生成逻辑，模拟实际场景。

## 📚 参考资料

- [FastMCP 文档](https://github.com/jlowin/fastmcp)
- [MCP 协议](https://modelcontextprotocol.io/)
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/)
- [主项目 README](../README.md)

---

**注意**: 当前版本返回模拟数据，生产环境需配置真实 API。
