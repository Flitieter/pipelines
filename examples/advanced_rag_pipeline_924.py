"""高级RAG Pipeline

这是一个完整的RAG（检索增强生成）pipeline，包含以下功能：
1. 问题改写：使用LLM优化和扩展用户查询
2. 混合检索：结合向量检索和BM25关键词检索
3. 重排序：使用cross-encoder对检索结果进行重新排序
4. 答案生成：基于检索到的相关文档生成最终答案

技术架构：
- 向量检索：使用sentence-transformers进行语义相似度检索
- 关键词检索：使用BM25算法进行精确匹配检索
- 混合检索：结合两种检索方式的优势
- 重排序：使用cross-encoder模型对检索结果重新排序
- 答案生成：基于OpenAI API生成高质量答案

支持的数据源：
- SQuAD数据集：标准问答数据集
- 本地文档：支持txt、md、json格式文件
- 自动分块：智能文档分块处理

requirements: chromadb, sentence-transformers, rank-bm25, numpy, openai, jieba, datasets, alayalite
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Any, Union, Generator, Iterator, Optional
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime

# =============================================================================
# 依赖包导入部分
# =============================================================================
# 采用安全导入策略，分别导入各个依赖包，避免一个失败影响全部
# 如果某个包导入失败，会设置为None，后续代码会检查并提供降级方案

# 初始化所有可选依赖为None
chromadb = None          # 向量数据库
SentenceTransformer = None  # 句子嵌入模型
CrossEncoder = None      # 交叉编码器重排序模型
BM25Okapi = None        # BM25关键词检索算法
openai = None           # OpenAI API客户端
jieba = None            # 中文分词工具
re = None               # 正则表达式模块
datasets = None         # HuggingFace数据集库
AlayaLiteClient = None  # AlayaLite向量数据库客户端
watchdog = None         # 文件监控库
threading = None        # 线程库
time = None             # 时间库

# 尝试导入ChromaDB - 用于向量存储和检索
try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    print(f"ChromaDB导入失败: {e}")
    chromadb = None

# 尝试导入AlayaLite客户端 - 用于向量存储和检索
try:
    # 与vector_db.py保持一致的导入方式
    from alayalite import Client
    AlayaLiteClient = Client  # 保持兼容性
    AlayaLiteConfig = None  # 使用简单的API，不需要配置类
except ImportError as e:
    print(f"AlayaLite导入失败: {e}")
    AlayaLiteClient = None

# 尝试导入Sentence Transformers - 用于文本嵌入和重排序
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError as e:
    print(f"Sentence Transformers导入失败: {e}")
    SentenceTransformer = None
    CrossEncoder = None

# 尝试导入BM25 - 用于关键词检索
try:
    from rank_bm25 import BM25Okapi
except ImportError as e:
    print(f"BM25导入失败: {e}")
    BM25Okapi = None

# 尝试导入OpenAI - 用于LLM调用
try:
    import openai
except ImportError as e:
    print(f"OpenAI导入失败: {e}")
    openai = None

# 尝试导入Jieba - 用于中文分词
try:
    import jieba
except ImportError as e:
    print(f"Jieba导入失败: {e}")
    jieba = None

# 尝试导入正则表达式模块
try:
    import re
except ImportError as e:
    print(f"Re导入失败: {e}")
    re = None

# 尝试导入HuggingFace Datasets - 用于加载标准数据集
try:
    import datasets
except ImportError as e:
    print(f"Datasets导入失败: {e}")
    datasets = None

# 尝试导入watchdog - 用于文件监控
try:
    import watchdog
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError as e:
    print(f"watchdog导入失败: {e}")
    watchdog = None

# 尝试导入threading - 用于线程管理
try:
    import threading
except ImportError as e:
    print(f"threading导入失败: {e}")
    threading = None

# 尝试导入time - 用于时间管理
try:
    import time
except ImportError as e:
    print(f"time导入失败: {e}")
    time = None

required_packages = []
if not chromadb and not AlayaLiteClient:
    required_packages.append("chromadb 或 alayalite")
if not SentenceTransformer:
    required_packages.append("sentence-transformers")
if not BM25Okapi:
    required_packages.append("rank-bm25")
if not openai:
    required_packages.append("openai")
if not jieba:
    required_packages.append("jieba")
if not re:
    required_packages.append("re")

if required_packages:
    print(f"部分依赖包缺失，请运行: pip install {' '.join(required_packages)}")


# =============================================================================
# 主要Pipeline类定义
# =============================================================================

class Pipeline:
    """高级RAG Pipeline类
    
    这是一个完整的RAG（检索增强生成）系统，实现了从文档索引到答案生成的完整流程。
    
    主要功能模块：
    1. 文档加载与预处理：支持多种数据源和格式
    2. 文档分块：智能分割长文档为可检索的块
    3. 向量索引：使用嵌入模型构建语义检索索引
    4. BM25索引：构建关键词检索索引
    5. 混合检索：结合语义和关键词检索
    6. 结果重排序：使用交叉编码器优化检索结果
    7. 答案生成：基于检索到的相关文档生成答案
    
    设计特点：
    - 模块化设计：各个组件可独立配置和替换
    - 容错机制：依赖包缺失时提供降级方案
    - 异步处理：支持高并发查询处理
    - 可配置性：丰富的配置参数满足不同需求
    """
    
    class Valves(BaseModel):
        """Pipeline配置参数
        
        使用Pydantic BaseModel实现类型安全的配置管理。
        所有配置参数都有默认值，用户可以根据需要进行调整。
        
        配置分类：
        - API配置：OpenAI相关设置
        - 检索配置：向量检索和BM25检索参数
        - 模型配置：嵌入模型和重排序模型设置
        - 数据配置：数据源路径和处理参数
        - 系统配置：调试模式和日志设置
        """
        # =================================================================
        # API配置 - OpenAI相关设置
        # =================================================================
        OPENAI_API_KEY: str = Field(
            default="sk-IcP6K2EsRybG1NTXehaaiw",
            description="OpenAI API密钥，用于调用LLM进行问题改写和答案生成"
        )
        OPENAI_API_BASE: str = Field(
            default="https://llmapi.paratera.com/v1",
            description="OpenAI API基础URL，支持自定义API端点"
        )
        MODEL_NAME: str = Field(
            default="DeepSeek-R1-0528",
            description="使用的大语言模型名称，用于问题改写和答案生成"
        )
        
        # =================================================================
        # 检索配置 - 控制检索过程中的文档数量
        # =================================================================
        TOP_K_RETRIEVAL: int = Field(
            default=10,
            description="初始检索阶段返回的文档数量，数量越多召回率越高但计算成本越大"
        )
        TOP_K_RERANK: int = Field(
            default=5,
            description="重排序后保留的文档数量，在精度和效率之间平衡"
        )
        TOP_K_FINAL: int = Field(
            default=3,
            description="最终用于生成答案的文档数量，影响答案质量和生成速度"
        )
        
        # =================================================================
        # 模型配置 - 嵌入和重排序模型设置
        # =================================================================
        EMBEDDING_MODEL: str = Field(
            default="Qwen/Qwen3-Embedding-0.6B",
            description="Qwen嵌入模型，用于将查询和文档转换为向量表示，支持多语言语义相似度计算"
        )
        RERANK_MODEL: str = Field(
            default="cross-encoder/ms-marco-MiniLM-L-2-v2",
            description="交叉编码器重排序模型，用于精确评估查询与文档的相关性"
        )
        
        # =================================================================
        # 检索质量控制 - 过滤和优化检索结果
        # =================================================================
        MIN_SIMILARITY_THRESHOLD: float = Field(
            default=0.2,
            description="最小相似度阈值，低于此值的结果将被过滤，提高结果相关性。针对AlayaLite调整为更低的阈值"
        )
        QUERY_EXPANSION: bool = Field(
            default=False,
            description="是否启用查询扩展，通过LLM改写查询以提高检索效果"
        )
        
        # =================================================================
        # 数据配置 - 文档存储和处理设置
        # =================================================================
        DATA_PATH: str = Field(
            default="./testdata",
            description="文档数据存储路径，支持txt、md、json等格式文件。如果使用vault，则为vault挂载的本地路径"
        )
        VAULT_ID: str = Field(
            default="ade202be-3052-4e22-8b53-0cb011c007f0",
            description="Vault ID，用于监控特定的vault文件夹"
        )
        VECTOR_DB_PATH: str = Field(
            default="./chroma_db",
            description="向量数据库存储路径，用于持久化向量索引"
        )
        USE_ALAYALITE: bool = Field(
            default=True,
            description="是否使用AlayaLite向量数据库"
        )
        ALAYALITE_URL: str = Field(
            default="http://localhost:8000",
            description="AlayaLite服务URL"
        )
        ALAYALITE_NAMESPACE: str = Field(
            default="rag_namespace",
            description="AlayaLite命名空间（注意：当前实现中此参数已不再使用，直接使用VAULT_ID作为collection名称）"
        )
        CHUNK_SIZE: int = Field(
            default=800,
            description="文档分块大小（字符数），影响检索粒度和上下文完整性"
        )
        CHUNK_OVERLAP: int = Field(
            default=100,
            description="文档分块重叠大小，确保重要信息不会在分块边界丢失"
        )
        
        # =================================================================
        # 混合检索权重 - 平衡语义检索和关键词检索
        # =================================================================
        VECTOR_WEIGHT: float = Field(
            default=0.7,
            description="向量检索权重，控制语义相似度在混合检索中的重要性"
        )
        BM25_WEIGHT: float = Field(
            default=0.3,
            description="BM25检索权重，控制关键词匹配在混合检索中的重要性"
        )
        
        # =================================================================
        # 答案生成配置 - 控制LLM生成行为
        # =================================================================
        MAX_TOKENS: int = Field(
            default=1000,
            description="生成答案的最大token数，控制答案长度"
        )
        TEMPERATURE: float = Field(
            default=0.1,
            description="生成温度，较低值产生更确定性的答案，较高值增加创造性"
        )
        
        # =================================================================
        # 系统配置 - 调试和日志设置
        # =================================================================
        DEBUG: bool = Field(
            default=True,
            description="是否启用调试模式，输出详细的执行日志"
        )
        
        # =================================================================
        # 文件监控配置
        # =================================================================
        ENABLE_FILE_MONITORING: bool = Field(
            default=True,
            description="是否启用vault文件夹监控"
        )
        MONITOR_INTERVAL: int = Field(
            default=10,
            description="文件监控轮询间隔（秒）"
        )
        USE_WATCHDOG: bool = Field(
            default=True,
            description="是否使用watchdog进行实时文件监控"
        )
        

    
    def __init__(self):
        """初始化Pipeline
        
        设置Pipeline的基本属性、配置参数、日志系统和核心组件。
        采用延迟初始化策略，在首次使用时才加载重型组件（如模型和数据库）。
        """
        # =================================================================
        # 基本属性设置 - Pipeline框架要求的标准属性
        # =================================================================
        self.type = "manifold"  # Pipeline类型，支持多种输入输出格式
        self.id = "advanced_rag_pipeline"  # 唯一标识符
        self.name = "Advanced RAG Pipeline"  # 显示名称
        self.valves = self.Valves()  # 配置参数实例
        
        # 框架兼容性属性 - 支持Open WebUI框架
        self.pipelines = [{
            "id": self.id,
            "name": "RAG Pipeline",
            "type": self.type
        }]
        
        # 模型端点支持 - 使Pipeline可以作为模型服务
        self.models = [{
            "id": self.id,
            "name": "RAG Pipeline",
            "object": "model",
            "created": 1640995200,
            "owned_by": "advanced-rag"
        }]
        
        # =================================================================
        # 日志系统设置 - 提供详细的执行跟踪
        # =================================================================
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # =================================================================
        # 核心组件初始化 - 延迟加载策略
        # =================================================================
        # 所有重型组件初始化为None，在首次使用时才加载，提高启动速度
        self.embedding_model = None    # 文本嵌入模型，用于向量化文档和查询
        self.rerank_model = None       # 重排序模型，用于精确评估相关性
        self.vector_store = None       # 向量数据库，存储文档向量
        self.vdb_controller = None     # AlayaLite控制器
        self.bm25_index = None         # BM25索引，用于关键词检索
        self.document_chunks = None    # 文档分块列表，存储原始文本
        self.openai_client = None      # OpenAI API客户端，用于LLM调用
        
        # =================================================================
        # 文件监控相关属性初始化
        # =================================================================
        self._file_status = None       # 存储文件状态记录
        self._last_monitor_time = 0    # 上次监控时间
        self._stop_event = None        # 停止监控的事件对象
        self._observer = None          # watchdog观察者实例
        self._monitor_thread = None    # 轮询监控线程
        
        # 系统状态标志
        self._initialized = False      # 标记是否已完成初始化
        
        # 文件监控相关属性
        self._monitor_thread = None     # 文件监控线程
        self._observer = None           # watchdog观察者
        self._stop_event = None         # 停止事件标志
        self._file_status = {}          # 记录文件修改时间和大小
        self._last_monitor_time = 0     # 上次监控时间戳
        
        # 配置更新锁，防止并发更新冲突
        self._config_lock = threading.Lock() if threading else None
    
    async def on_startup(self):
        """Pipeline启动时的初始化方法
        
        在系统启动时执行一次性初始化操作，包括：
        1. OpenAI客户端初始化
        2. 嵌入模型加载
        3. 重排序模型加载
        4. 数据加载和索引构建
        
        采用容错设计，即使某些组件初始化失败，系统仍能以降级模式运行。
        """
        try:
            if self.valves.DEBUG:
                self.logger.info("正在初始化Advanced RAG Pipeline...")
            
            # =================================================================
            # OpenAI客户端初始化 - 用于LLM调用
            # =================================================================
            if self.valves.OPENAI_API_KEY and openai is not None:
                self.openai_client = openai.AsyncOpenAI(
                    api_key=self.valves.OPENAI_API_KEY,
                    base_url=self.valves.OPENAI_API_BASE
                )
                if self.valves.DEBUG:
                    self.logger.info("OpenAI客户端初始化成功")
            elif self.valves.OPENAI_API_KEY and openai is None:
                self.logger.warning("OpenAI模块未安装，将无法使用LLM功能")
                self.openai_client = None
            else:
                self.logger.warning("未配置OpenAI API密钥，将无法使用LLM功能")
                self.openai_client = None
            
            # =================================================================
            # 嵌入模型初始化 - 用于文本向量化
            # =================================================================
            if SentenceTransformer:
                if self.valves.DEBUG:
                    self.logger.info(f"正在加载嵌入模型: {self.valves.EMBEDDING_MODEL}")
                self.embedding_model = SentenceTransformer(self.valves.EMBEDDING_MODEL)
                if self.valves.DEBUG:
                    self.logger.info(f"嵌入模型加载完成: {self.valves.EMBEDDING_MODEL}")
            else:
                self.logger.warning("SentenceTransformer模块未安装，将无法使用向量检索")
            
            # =================================================================
            # 重排序模型初始化 - 用于精确相关性评估（可选组件）
            # =================================================================
            try:
                if CrossEncoder:
                    if self.valves.DEBUG:
                        self.logger.info(f"正在加载重排序模型: {self.valves.RERANK_MODEL}")
                    self.rerank_model = CrossEncoder(self.valves.RERANK_MODEL)
                    if self.valves.DEBUG:
                        self.logger.info(f"重排序模型加载完成: {self.valves.RERANK_MODEL}")
                else:
                    self.logger.warning("CrossEncoder模块未安装，将使用简单重排序")
            except Exception as e:
                self.logger.warning(f"重排序模型加载失败，将使用简单重排序: {str(e)}")
                self.rerank_model = None
            
            # =================================================================
            # 向量数据库初始化 - 连接.alayabox中的向量数据库存储（向量索引）
            # =================================================================
            try:
                # 构建.alayabox中的vault路径
                alayabox_path = os.path.join(os.path.expanduser("~"), ".alayabox")
                vault_path = os.path.join(alayabox_path, self.valves.VAULT_ID)
                
                # 使用AlayaLite向量数据库
                if self.valves.USE_ALAYALITE and AlayaLiteClient:
                    if self.valves.DEBUG:
                        self.logger.info(f"正在初始化AlayaLite，连接到: {vault_path}")
                    
                    try:
                        # 初始化AlayaLite客户端，与vector_db.py保持完全一致
                        # ALAYALITE_DATA_PATH = Path.home() / '.alayabox'
                        alayabox_path = os.path.join(os.path.expanduser("~"), ".alayabox")
                        self.vdb_controller = AlayaLiteClient(url=alayabox_path)
                        
                        # 将控制器同时设置为vector_store以兼容现有代码
                        self.vector_store = self.vdb_controller
                        
                        if self.valves.DEBUG:
                            self.logger.info(f"AlayaLite初始化完成，路径: {alayabox_path}")
                            self.logger.info(f"AlayaLite客户端状态: {'已初始化'}")
                            self.logger.info(f"向量检索将使用: {'AlayaLite'}")
                            self.logger.info(f"将使用Vault ID '{self.valves.VAULT_ID}'作为collection名称")
                    except Exception as e:
                        self.logger.error(f"AlayaLite初始化失败: {str(e)}")
                        self.vdb_controller = None
                        self.vector_store = None
                
                # 如果AlayaLite不可用，尝试使用ChromaDB但从.alayabox加载
                if not self.vector_store and chromadb:
                    if self.valves.DEBUG:
                        self.logger.info(f"正在初始化ChromaDB，从.alayabox加载: {alayabox_path}")
                    
                    # 检查vault路径是否存在
                    if os.path.exists(alayabox_path):
                        try:
                            chroma_client = chromadb.PersistentClient(path=alayabox_path)
                            collection_name = "rag_documents"
                            
                            # 尝试加载现有集合
                            self.vector_store = chroma_client.get_collection(name=collection_name)
                            if self.valves.DEBUG:
                                self.logger.info(f"从.alayabox加载向量存储成功: {collection_name}")
                        except Exception as e:
                            self.logger.warning(f"从.alayabox加载向量存储失败，将创建新集合: {str(e)}")
                            # 创建新集合，使用余弦相似度
                            self.vector_store = chroma_client.create_collection(
                                name=collection_name,
                                metadata={"hnsw:space": "cosine"}
                            )
                            if self.valves.DEBUG:
                                self.logger.info(f"在.alayabox创建新向量存储: {collection_name}")
                    else:
                        self.logger.warning(f"vault路径不存在: {vault_path}")
                
                elif not self.vector_store:
                    self.logger.warning("无法连接到向量数据库，将无法使用向量检索功能")
                
            except Exception as e:
                self.logger.error(f"向量数据库初始化失败: {str(e)}")
            
            # =================================================================
            # 初始化完成标记
            # =================================================================
            self._initialized = True
            if self.valves.DEBUG:
                self.logger.info("Advanced RAG Pipeline初始化完成")
                
            # 启动文件监控
            if self.valves.ENABLE_FILE_MONITORING and self.valves.VAULT_ID:
                if threading or watchdog:
                    try:
                        await self._start_file_monitoring()
                    except Exception as e:
                        self.logger.warning(f"文件监控启动失败，将继续使用其他功能: {str(e)}")
                else:
                    self.logger.warning("threading和watchdog模块均未安装，无法启动文件监控")
                
        except Exception as e:
            self.logger.error(f"Pipeline初始化失败: {str(e)}")
            raise
    
    async def on_shutdown(self):
        """Pipeline关闭时的资源清理方法
        
        在系统关闭时执行清理操作，确保资源正确释放：
        1. 关闭数据库连接
        2. 清理模型缓存
        3. 释放内存资源
        4. 停止文件监控
        """
        if self.valves.DEBUG:
            self.logger.info("正在关闭Advanced RAG Pipeline...")
            
        # 停止文件监控
        if self.valves.ENABLE_FILE_MONITORING:
            await self._stop_file_monitoring()
    
    def update_config(self, vault_id: str, data_path: str) -> dict:
        """动态更新Pipeline配置
        
        参数:
            vault_id: 新的vault ID
            data_path: 新的数据路径
            
        返回:
            dict: 更新结果状态
        """
        try:
            if self._config_lock:
                with self._config_lock:
                    return self._update_config_internal(vault_id, data_path)
            else:
                return self._update_config_internal(vault_id, data_path)
        except Exception as e:
            self.logger.error(f"配置更新失败: {str(e)}")
            return {
                "success": False,
                "message": f"配置更新失败: {str(e)}"
            }
    
    def _update_config_internal(self, vault_id: str, data_path: str) -> dict:
        """内部配置更新逻辑"""
        old_vault_id = self.valves.VAULT_ID
        old_data_path = self.valves.DATA_PATH
        
        # 更新配置
        self.valves.VAULT_ID = vault_id
        self.valves.DATA_PATH = data_path
        
        # 重置初始化状态，强制重新初始化
        self._initialized = False
        
        # 清理旧的组件
        self.vector_store = None
        self.vdb_controller = None
        self.bm25_index = None
        self.document_chunks = None
        
        # 停止文件监控
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join()
                self._observer = None
            except Exception as e:
                self.logger.warning(f"停止文件监控失败: {str(e)}")
        
        if self._monitor_thread:
            try:
                if self._stop_event:
                    self._stop_event.set()
                self._monitor_thread.join(timeout=5)
                self._monitor_thread = None
            except Exception as e:
                self.logger.warning(f"停止监控线程失败: {str(e)}")
        
        self.logger.info(f"配置已更新: VAULT_ID {old_vault_id} -> {vault_id}, DATA_PATH {old_data_path} -> {data_path}")
        
        return {
            "success": True,
            "message": "配置更新成功",
            "old_config": {
                "vault_id": old_vault_id,
                "data_path": old_data_path
            },
            "new_config": {
                "vault_id": vault_id,
                "data_path": data_path
            }
        }
    
    # =============================================================================
    # 文件监控相关方法
    # =============================================================================
    async def _start_file_monitoring(self):
        """启动DATA_PATH目录监控，实时更新BM25索引"""
        try:
            self.logger.info(f"正在启动对 Vault ID: {self.valves.VAULT_ID} 的文件监控，路径: {self.valves.DATA_PATH}")
            # 直接使用DATA_PATH作为监控路径
            monitor_path = self.valves.DATA_PATH
            
            if not os.path.exists(monitor_path):
                self.logger.warning(f"监控路径不存在: {monitor_path}")
                # 创建监控目录
                os.makedirs(monitor_path, exist_ok=True)
                self.logger.info(f"已创建监控目录: {monitor_path}")
                
            # 初始化文件状态记录
            self._file_status = self._get_current_file_status(monitor_path)
            self._last_monitor_time = time.time() if time else 0
            self._stop_event = threading.Event() if threading else None
            
            # 根据配置选择监控方式
            if self.valves.USE_WATCHDOG and watchdog and VaultFileHandler:
                # 使用watchdog进行实时监控
                self._observer = watchdog.observers.Observer()
                event_handler = VaultFileHandler(self)
                self._observer.schedule(event_handler, monitor_path, recursive=True)
                self._observer.start()
                if self.valves.DEBUG:
                    self.logger.info(f"使用watchdog启动实时文件监控: {monitor_path}")
            elif self.valves.USE_WATCHDOG and (not watchdog or not VaultFileHandler):
                self.logger.warning("watchdog监控不可用，尝试使用轮询方式")
            elif threading:
                # 使用轮询方式监控
                self._monitor_thread = threading.Thread(
                    target=self._poll_files,
                    args=(monitor_path,),
                    daemon=True
                )
                self._monitor_thread.start()
                if self.valves.DEBUG:
                    self.logger.info(f"使用轮询方式启动文件监控: {monitor_path}，间隔 {self.valves.MONITOR_INTERVAL} 秒")
            else:
                self.logger.warning("无法启动文件监控，缺少必要的依赖库")
                
        except Exception as e:
            self.logger.error(f"启动文件监控失败: {str(e)}")
            
    async def _stop_file_monitoring(self):
        """停止文件监控"""
        try:
            if self._stop_event:
                self._stop_event.set()
                
            if self._observer and self.valves.USE_WATCHDOG:
                self._observer.stop()
                self._observer.join(timeout=5)
                self._observer = None
                
            if self._monitor_thread and not self.valves.USE_WATCHDOG:
                self._monitor_thread.join(timeout=5)
                self._monitor_thread = None
                
            if self.valves.DEBUG:
                self.logger.info("文件监控已停止")
                
        except Exception as e:
            self.logger.error(f"停止文件监控失败: {str(e)}")
            
    def _poll_files(self, directory):
        """轮询方式监控文件变化"""
        try:
            while self._stop_event and not self._stop_event.is_set():
                current_time = time.time() if time else 0
                # 检查是否达到监控间隔
                if current_time - self._last_monitor_time >= self.valves.MONITOR_INTERVAL:
                    self._last_monitor_time = current_time
                    
                    # 异步调用检查文件变化
                    if asyncio:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self._check_file_changes(directory))
                        loop.close()
                    else:
                        self.logger.warning("asyncio不可用，无法检查文件变化")
                    
                # 短暂休眠避免CPU占用过高
                if time:
                    time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"文件轮询监控失败: {str(e)}")
            
    async def _check_file_changes(self, directory):
        """检查目录中的文件变化"""
        try:
            # 获取当前文件状态
            current_status = self._get_current_file_status(directory)
            
            # 检查新增、修改或删除的文件
            has_changes = False
            
            # 检查新增或修改的文件
            for file_path, status in current_status.items():
                if file_path not in self._file_status or self._file_status[file_path] != status:
                    has_changes = True
                    break
                    
            # 检查删除的文件
            for file_path in self._file_status:
                if file_path not in current_status:
                    has_changes = True
                    break
                    
            # 如果有文件变化，更新BM25索引
            if has_changes:
                self._file_status = current_status
                await self._update_bm25_index()
                
        except Exception as e:
            self.logger.error(f"检查文件变化失败: {str(e)}")
            
    async def _update_bm25_index(self):
        """更新BM25索引"""
        try:
            if self.valves.DEBUG:
                self.logger.info("开始更新BM25索引...")
                
            # 重新加载文档并分块
            documents = await self._load_documents_for_indexing()
            if documents:
                self.document_chunks = self._chunk_documents_for_indexing(documents)
                # 重新构建BM25索引
                self._build_bm25_index()
                
                if self.valves.DEBUG:
                    self.logger.info(f"BM25索引更新完成，包含 {len(self.document_chunks)} 个文档块")
            else:
                self.logger.warning("没有找到文档，BM25索引未更新")
                
        except Exception as e:
            self.logger.error(f"更新BM25索引失败: {str(e)}")
            
    def _get_current_file_status(self, directory):
        """获取目录中文件的当前状态（修改时间和大小）"""
        file_status = {}
        try:
            supported_extensions = {'.txt', '.md', '.json'}
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    if file_ext in supported_extensions:
                        try:
                            # 获取文件修改时间和大小
                            stat = os.stat(file_path)
                            file_status[file_path] = (stat.st_mtime, stat.st_size)
                        except Exception as e:
                            self.logger.warning(f"获取文件状态失败 {file_path}: {str(e)}")
                            
        except Exception as e:
            self.logger.error(f"获取文件状态失败: {str(e)}")
            
        return file_status
    
    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Pipeline主入口函数
        
        这是Pipeline的核心接口，处理来自用户的查询请求。
        
        参数:
            user_message: 用户输入的查询文本
            model_id: 模型标识符（用于兼容性）
            messages: 对话历史消息列表
            body: 请求体（包含额外参数）
            
        返回:
            str: 生成的答案文本
            
        处理流程:
            1. 接收用户查询
            2. 创建异步事件循环
            3. 调用异步处理逻辑
            4. 返回处理结果
        """
        try:
            if self.valves.DEBUG:
                self.logger.info(f"收到查询: {user_message}")
            
            # =================================================================
            # 异步事件循环管理 - 确保异步操作正确执行
            # =================================================================
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._process_query(user_message, messages)
                )
                return result
            finally:
                loop.close()  # 确保事件循环正确关闭
                
        except Exception as e:
            self.logger.error(f"Pipeline处理失败: {str(e)}")
            return f"抱歉，处理您的查询时出现错误: {str(e)}"
    
    async def _process_query(self, user_message: str, messages: List[dict]) -> str:
        """异步处理查询的核心逻辑
        
        实现完整的RAG流程：
        1. 索引初始化检查
        2. 查询改写（可选）
        3. 混合检索（向量+BM25）
        4. 结果重排序
        5. 答案生成
        
        参数:
            user_message: 用户查询文本
            messages: 对话历史
            
        返回:
            str: 生成的答案
        """
        try:
            if self.valves.DEBUG:
                self.logger.info(f"开始处理查询: {user_message}")
            
            # =================================================================
            # 索引初始化检查 - 确保检索系统就绪
            # =================================================================
            if not self._is_initialized():
                if self.valves.DEBUG:
                    self.logger.info("检测到索引未初始化，开始初始化...")
                await self._initialize_indexes()
            
            # 1. 问题改写（可选）
            if self.valves.QUERY_EXPANSION:
                rewritten_queries = await self._rewrite_query(user_message, messages)
                if self.valves.DEBUG:
                    self.logger.info(f"改写后的查询: {rewritten_queries}")
            else:
                rewritten_queries = [user_message]  # 直接使用原始查询
                if self.valves.DEBUG:
                    self.logger.info("查询扩展已禁用，使用原始查询")
            
            # 2. 混合检索
            retrieved_docs = await self._hybrid_retrieve(rewritten_queries)
            if self.valves.DEBUG:
                self.logger.info(f"检索到 {len(retrieved_docs)} 个文档")
            
            # 3. 重排序
            ranked_docs = await self._rerank_documents(user_message, retrieved_docs)
            if self.valves.DEBUG:
                self.logger.info(f"重排序后保留 {len(ranked_docs)} 个文档")
            
            # 4. 生成最终回答
            answer = await self._generate_answer(user_message, ranked_docs, messages)
            
            return answer
            
        except Exception as e:
            self.logger.error(f"处理查询失败: {str(e)}")
            raise
    
    def _is_initialized(self) -> bool:
        """检查索引是否已初始化"""
        return (
            self.vector_store is not None and 
            self.bm25_index is not None and 
            self.document_chunks is not None and 
            len(self.document_chunks) > 0
        )
    
    async def _initialize_indexes(self):
        """初始化向量索引和BM25索引"""
        try:
            if self.valves.DEBUG:
                self.logger.info("开始初始化索引...")
            
            # 加载文档、分块处理和构建BM25索引
            await self._load_and_build_document_indexes()
            
            # 构建向量索引
            await self._build_vector_index()
            
            if self.valves.DEBUG:
                self.logger.info("索引初始化完成")
                
        except Exception as e:
            self.logger.error(f"索引初始化失败: {str(e)}")
            
    async def _load_and_build_document_indexes(self):
        """加载文档、分块处理并构建BM25索引"""
        try:
            # 加载文档
            documents = await self._load_documents_for_indexing()
            if not documents:
                self.logger.warning("未找到任何文档，索引初始化失败")
                return
            
            # 分块处理文档
            self.document_chunks = self._chunk_documents_for_indexing(documents)
            if self.valves.DEBUG:
                self.logger.info(f"文档分块完成，共 {len(self.document_chunks)} 个块")
            
            # 构建BM25索引
            self._build_bm25_index()
            
        except Exception as e:
            self.logger.error(f"加载文档和构建索引失败: {str(e)}")
    
    async def _load_documents_for_indexing(self) -> List[Dict]:
        """从数据目录加载文档用于索引构建"""
        try:
            documents = []
            
            # 从数据目录加载文档
            file_docs = await self._load_documents_from_files()
            documents.extend(file_docs)
            if self.valves.DEBUG:
                self.logger.info(f"从文件加载了 {len(file_docs)} 个文档")
            
            if self.valves.DEBUG:
                self.logger.info(f"总共加载了 {len(documents)} 个文档")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"加载文档失败: {str(e)}")
            return []
    

    
    async def _load_documents_from_files(self) -> List[Dict]:
        """从DATA_PATH直接加载文档文件"""
        try:
            documents = []
            
            # 直接使用DATA_PATH作为文档加载路径
            data_path = self.valves.DATA_PATH
            if self.valves.DEBUG:
                self.logger.info(f"使用数据目录: {data_path}")
            
            if not os.path.exists(data_path):
                self.logger.warning(f"数据目录不存在: {data_path}")
                return documents
            
            # 支持的文件格式
            supported_extensions = {'.txt', '.md', '.json'}
            
            for root, dirs, files in os.walk(data_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    if file_ext in supported_extensions:
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if content:
                                    # 添加vault相关的元数据
                                    metadata = {
                                        'source': file_path,
                                        'filename': file,
                                        'extension': file_ext
                                    }
                                    
                                    # 如果使用vault，添加vault_id信息
                                    if self.valves.VAULT_ID:
                                        metadata['vault_id'] = self.valves.VAULT_ID
                                    
                                    documents.append({
                                        'content': content,
                                        'metadata': metadata
                                    })
                        except Exception as e:
                            self.logger.warning(f"读取文件失败 {file_path}: {str(e)}")
            
            return documents
            
        except Exception as e:
            self.logger.error(f"从文件加载文档失败: {str(e)}")
            return []
    
    def _chunk_documents_for_indexing(self, documents: List[Dict]) -> List[Dict]:
        """将文档分块用于索引构建"""
        try:
            chunks = []
            
            for doc in documents:
                content = doc['content']
                metadata = doc['metadata']
                
                # 简单的分块策略：按段落分割
                paragraphs = content.split('\n\n')
                
                current_chunk = ""
                chunk_index = 0
                
                for paragraph in paragraphs:
                    paragraph = paragraph.strip()
                    if not paragraph:
                        continue
                    
                    # 如果当前块加上新段落超过最大长度，保存当前块并开始新块
                    if len(current_chunk) + len(paragraph) > self.valves.CHUNK_SIZE:
                        if current_chunk:
                            chunks.append({
                                'content': current_chunk.strip(),
                                'metadata': {
                                    **metadata,
                                    'chunk_index': chunk_index,
                                    'chunk_id': f"{metadata['filename']}_{chunk_index}"
                                }
                            })
                            chunk_index += 1
                            current_chunk = paragraph
                        else:
                            # 单个段落就超过最大长度，直接作为一个块
                            chunks.append({
                                'content': paragraph,
                                'metadata': {
                                    **metadata,
                                    'chunk_index': chunk_index,
                                    'chunk_id': f"{metadata['filename']}_{chunk_index}"
                                }
                            })
                            chunk_index += 1
                    else:
                        if current_chunk:
                            current_chunk += "\n\n" + paragraph
                        else:
                            current_chunk = paragraph
                
                # 保存最后一个块
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': {
                            **metadata,
                            'chunk_index': chunk_index,
                            'chunk_id': f"{metadata['filename']}_{chunk_index}"
                        }
                    })
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"文档分块失败: {str(e)}")
            return []
    
    async def _build_vector_index(self):
        """构建向量索引并存储到.alayabox中的vaultid文件夹"""
        try:
            # 检查向量存储是否已初始化
            if not self.vector_store:
                if self.valves.DEBUG:
                    self.logger.warning("向量存储未初始化，跳过索引构建")
                return
            
            # 检查是否需要构建索引（如果文档块为空，可能已有向量数据）
            if not self.document_chunks:
                if self.valves.DEBUG:
                    self.logger.info("文档块为空，假设.alayabox中已有向量数据")
                return
            
            # 检查.alayabox下是否已有对应vault_id的collection
            if self.valves.USE_ALAYALITE and self.vdb_controller:
                collection_name = self.valves.VAULT_ID
                try:
                    # 尝试获取已存在的collection
                    collection = self.vdb_controller.get_collection(collection_name)
                    # 只要collection存在就返回
                    if collection:
                        if self.valves.DEBUG:
                            self.logger.info(f"找到已存在的collection: {collection_name}，跳过向量索引初始化")
                            # 添加debug信息，记录collection名称
                            self.logger.debug(f"Debug: 已确认存在vault_id对应的collection: {collection_name}")
                        return
                except Exception as e:
                    # 如果collection不存在，会抛出异常，此时继续构建索引
                    if self.valves.DEBUG:
                        self.logger.info(f"collection {collection_name}不存在或无法访问，将创建新的索引")
                        self.logger.debug(f"Debug: collection访问异常: {str(e)}")
            
            # 提取文档内容
            texts = [chunk['content'] for chunk in self.document_chunks]
            metadatas = [chunk['metadata'] for chunk in self.document_chunks]
            
            # 批处理参数
            batch_size = 1000  # 每批处理1000个文档，避免超出模型限制
            total_docs = len(texts)
            
            if self.valves.DEBUG:
                self.logger.info(f"开始向.alayabox中的向量数据库添加文档，总文档数: {total_docs}，批处理大小: {batch_size}")
            
            # 分批处理文档
            for batch_start in range(0, total_docs, batch_size):
                batch_end = min(batch_start + batch_size, total_docs)
                
                # 当前批次的数据
                batch_texts = texts[batch_start:batch_end]
                batch_metadatas = metadatas[batch_start:batch_end]
                batch_ids = [f"doc_{i}" for i in range(batch_start, batch_end)]
                
                if self.valves.DEBUG:
                    self.logger.info(f"处理批次 {batch_start//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}，文档范围: {batch_start}-{batch_end-1}")
                
                # 生成当前批次的嵌入
                batch_embeddings = self.embedding_model.encode(batch_texts, show_progress_bar=True)
                
                # 添加到向量存储
                if self.valves.USE_ALAYALITE and self.vdb_controller:
                    # 使用AlayaLite API添加到.alayabox，与vector_db.py保持一致
                    # 使用vault_id作为collection名称
                    collection_name = self.valves.VAULT_ID
                    collection = self.vdb_controller.get_or_create_collection(collection_name)
                    
                    # 准备数据格式以匹配AlayaLite的API
                    docs_to_insert = list(zip(
                        batch_ids,
                        batch_texts,
                        [emb.tolist() for emb in batch_embeddings],
                        batch_metadatas
                    ))
                    
                    # 插入数据
                    collection.insert(docs_to_insert)
                    
                    # 保存集合
                    self.vdb_controller.save_collection(collection_name)
                    
                    if self.valves.DEBUG:
                        self.logger.info(f"文档已添加到collection: {collection_name}")
                elif self.vector_store:
                    # 使用ChromaDB API
                    self.vector_store.add(
                        embeddings=batch_embeddings.tolist(),
                        documents=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                
                if self.valves.DEBUG:
                    self.logger.info(f"批次 {batch_start//batch_size + 1} 处理完成，包含 {len(batch_texts)} 个文档")
            
            if self.valves.DEBUG:
                self.logger.info(f"向量索引构建完成，总共处理 {total_docs} 个文档")
                
        except Exception as e:
            self.logger.error(f"构建向量索引失败: {str(e)}")
    
    def _tokenize_text(self, text: str) -> List[str]:
        """智能分词函数，支持中英文混合文本"""
        try:
            if not text:
                return []
            
            # 清理文本
            text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not jieba:
                # 如果jieba不可用，使用简单分词
                return text.lower().split()
            
            # 使用jieba进行中文分词
            tokens = []
            words = jieba.lcut(text)
            
            for word in words:
                word = word.strip().lower()
                if len(word) > 1:  # 过滤单字符和空字符
                    tokens.append(word)
            
            return tokens
            
        except Exception as e:
            self.logger.warning(f"分词失败，使用简单分词: {str(e)}")
            return text.lower().split()
    
    def _build_bm25_index(self):
        """构建BM25索引"""
        try:
            if not self.document_chunks:
                return
            
            # 准备文档文本用于BM25，使用改进的分词
            corpus = []
            for chunk in self.document_chunks:
                tokens = self._tokenize_text(chunk['content'])
                corpus.append(tokens)
            
            # 创建BM25索引
            self.bm25_index = BM25Okapi(corpus)
            
            if self.valves.DEBUG:
                self.logger.info(f"BM25索引构建完成，包含 {len(corpus)} 个文档")
                
        except Exception as e:
            self.logger.error(f"构建BM25索引失败: {str(e)}")
            self.bm25_index = None
    

    
    async def _rewrite_query(self, query: str, messages: List[dict]) -> List[str]:
        """问题改写模块 - 使用LLM进行查询优化和扩展"""
        try:
            if not self.openai_client:
                # 如果没有OpenAI客户端，只返回原查询
                return [query]
            
            # 构建对话历史上下文
            context = ""
            if messages and len(messages) > 1:
                recent_messages = messages[-3:]  # 获取最近3条消息
                for msg in recent_messages:
                    if msg.get('role') == 'user':
                        context += f"用户: {msg.get('content', '')}\n"
                    elif msg.get('role') == 'assistant':
                        context += f"助手: {msg.get('content', '')}\n"
            
            # 构建改写提示词 - 针对SQuAD维基百科数据优化，生成单个英文查询
            system_prompt = """You are a professional query optimization expert. Your task is to rewrite user queries into a more precise and specific English search query for Wikipedia knowledge retrieval.

IMPORTANT: You MUST return ONLY a valid JSON object in this exact format: {"query": "your_rewritten_query"}
Do NOT include any explanations, comments, or additional text outside the JSON.

Query rewriting rules:
1. If the original query is already in good English and specific, expand it with relevant keywords
2. If the query is vague, make it more specific by adding context
3. For person queries: add full name, profession, and key achievements
4. For concept queries: add definition-related terms
5. For historical queries: add time period and key events
6. Always use encyclopedic terminology
7. The rewritten query should be significantly different from the original

Examples:
Original: "Who is Beethoven?"
Response: {"query": "Ludwig van Beethoven German composer classical music symphonies piano sonatas"}

Original: "Beyoncé ancestry"
Response: {"query": "Beyoncé Knowles family genealogy Acadian heritage Joseph Broussard descendant"}

Original: "Alaska facts"
Response: {"query": "Alaska United States largest state geography climate population history"}

Now rewrite the following query:"""
            
            user_prompt = f"""Original query: {query}

Rewrite this query following the rules above. Return only the JSON response:"""
            
            # 调用轻量级LLM进行查询改写
            response = await self.openai_client.chat.completions.create(
                model=self.valves.QUERY_REWRITE_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,  # 降低温度以获得更一致的结果
                max_tokens=300
            )
            
            # 解析响应
            response_text = response.choices[0].message.content.strip()
            
            try:
                # 尝试从响应中提取JSON部分
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start != -1 and json_end > json_start:
                    json_part = response_text[json_start:json_end]
                    result = json.loads(json_part)
                    rewritten_query = result.get('query', '')
                else:
                    # 如果没有找到JSON格式，尝试直接解析整个响应
                    result = json.loads(response_text)
                    rewritten_query = result.get('query', '')
                
                # 确保查询质量
                valid_queries = []
                if isinstance(rewritten_query, str) and len(rewritten_query.strip()) > 2:
                    # 检查重写查询是否与原查询不同
                    if rewritten_query.strip().lower() != query.strip().lower():
                        valid_queries.append(rewritten_query.strip())
                        if self.valves.DEBUG:
                            self.logger.info(f"查询重写成功: {query} -> {rewritten_query.strip()}")
                    else:
                        if self.valves.DEBUG:
                            self.logger.warning(f"重写查询与原查询相同，跳过重写结果")
                
                # 始终包含原始查询，确保原始问题和重写后的问题都用于检索
                if query not in valid_queries:
                    valid_queries.append(query)
                
                return valid_queries  # 返回原始查询和重写后的查询
                
            except json.JSONDecodeError:
                # JSON解析失败，尝试提取引号内的内容作为查询
                if self.valves.DEBUG:
                    self.logger.warning(f"JSON解析失败，原始响应: {response_text}")
                
                # 尝试从响应中提取有用的查询内容
                import re
                # 查找引号内的内容
                quoted_matches = re.findall(r'"([^"]+)"', response_text)
                if quoted_matches:
                    for match in quoted_matches:
                        if len(match.strip()) > 10 and match.strip().lower() != query.strip().lower():
                            if self.valves.DEBUG:
                                self.logger.info(f"从响应中提取到查询: {match.strip()}")
                            return [match.strip(), query]
                
                return [query]
            
        except Exception as e:
            self.logger.error(f"查询改写失败: {str(e)}")
            # 返回原查询作为备选
            return [query]
    
    async def _hybrid_retrieve(self, queries: List[str]) -> List[Dict]:
        """混合检索模块 - 结合向量检索和BM25关键词检索
        
        这是RAG系统的核心检索模块，采用混合检索策略：
        1. 向量检索：基于语义相似度，擅长理解查询意图
        2. BM25检索：基于关键词匹配，擅长精确匹配
        3. 结果融合：按权重合并两种检索结果
        
        参数:
            queries: 查询列表（包含原查询和改写查询）
            
        返回:
            List[Dict]: 检索到的文档列表，包含文档内容和相关性分数
            
        优势:
            - 语义检索捕获深层含义
            - 关键词检索确保精确匹配
            - 混合策略提高召回率和准确率
        """
        try:
            all_results = []
            
            for query in queries:
                if self.valves.DEBUG:
                    self.logger.info(f"正在检索查询: {query}")
                
                # 1. 向量检索
                vector_results = await self._vector_search(query)
                
                # 2. BM25关键词检索
                bm25_results = await self._bm25_search(query)
                
                # 3. 合并结果
                combined_results = self._merge_search_results(vector_results, bm25_results, query)
                all_results.extend(combined_results)
            
            # 4. 去重和排序
            final_results = self._deduplicate_and_rank(all_results)
            
            if self.valves.DEBUG:
                self.logger.info(f"混合检索返回 {len(final_results)} 个结果")
            
            return final_results[:self.valves.TOP_K_RETRIEVAL]
            
        except Exception as e:
            self.logger.error(f"混合检索失败: {str(e)}")
            return []
    
    async def _vector_search(self, query: str) -> List[Dict]:
        """向量检索 - 基于语义相似度的文档检索
        
        使用预训练的嵌入模型将查询和文档转换为向量表示，
        通过计算向量间的余弦相似度来找到语义相关的文档。
        
        参数:
            query: 用户查询文本
            
        返回:
            List[Dict]: 按相似度排序的文档列表
            
        工作流程:
            1. 将查询文本编码为向量
            2. 在向量数据库中搜索相似向量
            3. 计算相似度分数
            4. 应用相似度阈值过滤
            5. 返回格式化结果
        """
        try:
            if not self.vector_store:
                if self.valves.DEBUG:
                    self.logger.warning("向量存储未初始化")
                return []
            
            # =================================================================
            # 查询向量化 - 将文本转换为数值向量
            # =================================================================
            try:
                query_embedding = self.embedding_model.encode([query])
                if self.valves.DEBUG:
                    self.logger.debug(f"生成的查询向量维度: {len(query_embedding[0])}")
            except Exception as e:
                self.logger.error(f"查询向量化失败: {str(e)}")
                return []
            
            # =================================================================
            # 向量相似度搜索 - 在高维空间中找到最相似的文档
            # =================================================================
            search_k = min(self.valves.TOP_K_RETRIEVAL * 2, 20)  # 搜索更多结果用于后续过滤
            
            vector_results = []
            
            if self.valves.USE_ALAYALITE and self.vdb_controller:
                # 使用AlayaLite API进行查询，直接使用Client实例的正确方法
                if self.valves.DEBUG:
                    self.logger.info(f"使用AlayaLite执行查询，查询向量维度: {len(query_embedding[0])}")
                
                try:
                    # 直接使用Client实例的方法，与vector_db.py中的实现保持一致
                    collection = self.vdb_controller.get_collection(self.valves.VAULT_ID)
                    
                    if collection:
                        # 使用batch_query方法执行查询
                        result = collection.batch_query([query_embedding[0].tolist()], limit=search_k)
                        
                        if result and isinstance(result, dict):
                            # 添加调试信息以了解结果结构
                            if self.valves.DEBUG:
                                self.logger.info(f"AlayaLite查询结果键: {list(result.keys())}")
                                for key in ['id', 'document', 'metadata', 'distance']:
                                    if key in result:
                                        value = result[key]
                                        self.logger.info(f"结果[{key}]类型: {type(value)}, 长度: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                                        if hasattr(value, '__len__') and len(value) > 0:
                                            self.logger.info(f"结果[{key}][0]类型: {type(value[0])}")
                            
                            # 检查结果是否包含预期的键
                            if all(key in result for key in ['id', 'document', 'metadata', 'distance']):
                                # 处理嵌套列表结构：result['id'][0]是实际的数据列表
                                ids = result['id'][0] if isinstance(result['id'], list) and len(result['id']) > 0 else []
                                docs = result['document'][0] if isinstance(result['document'], list) and len(result['document']) > 0 else []
                                metadatas = result['metadata'][0] if isinstance(result['metadata'], list) and len(result['metadata']) > 0 else []
                                distances = result['distance'][0] if isinstance(result['distance'], list) and len(result['distance']) > 0 else []
                                

                                
                                # 处理结果格式
                                for i, (doc_id, doc, metadata, distance) in enumerate(zip(
                                    ids,
                                    docs,
                                    metadatas,
                                    distances
                                )):
                                    # 处理distance可能是列表或其他类型的情况
                                    actual_distance = distance
                                    
                                    # 递归处理嵌套列表的情况
                                    while isinstance(actual_distance, (list, tuple)) and len(actual_distance) > 0:
                                        actual_distance = actual_distance[0]
                                    
                                    # 确保actual_distance是数值类型
                                    try:
                                        actual_distance = float(actual_distance)
                                    except (ValueError, TypeError) as e:
                                        # 如果无法转换为浮点数，记录详细错误信息并跳过此结果
                                        self.logger.warning(f"距离值类型转换失败: distance={distance}, type={type(distance)}, error={str(e)}")
                                        continue
                                    

                                    
                                    # 确保距离值在合理范围内（放宽限制以适应不同的距离度量）
                                    if actual_distance < 0 or actual_distance > 10:  # 放宽上限从2到10
                                        self.logger.warning(f"距离值超出合理范围: {actual_distance}，跳过此结果")
                                        continue
                                    
                                    # 使用更鲁棒的相似度转换公式，适应AlayaLite的距离度量
                                    # AlayaLite的距离值通常在0.9-1.1范围内，需要特殊处理
                                    # 使用指数衰减函数，确保相似度在合理范围内
                                    similarity = 1.0 / (1.0 + actual_distance)  # 统一使用指数衰减函数
                                    

                                      
                                    # 应用相似度阈值过滤低质量结果
                                    if similarity >= self.valves.MIN_SIMILARITY_THRESHOLD:
                                        if self.valves.DEBUG and i < 3:  # 只打印前3个结果
                                            self.logger.debug(f"AlayaLite检索结果 {i+1}: 相似度={similarity:.4f}, ID={doc_id}")
                                     
                                        # 确保content是字符串类型
                                        try:
                                            # 安全处理doc内容
                                            content_str = str(doc) if doc is not None else ''
                                            
                                            # 增强metadata处理逻辑，修复嵌套列表和字典转换错误
                                            metadata_dict = {}
                                            if metadata is not None:
                                                try:
                                                    # 检查是否是嵌套列表结构（常见于AlayaLite返回格式）
                                                    if isinstance(metadata, (list, tuple)) and len(metadata) > 0 and isinstance(metadata[0], (list, tuple)):
                                                        # 取嵌套列表的第一个元素作为实际metadata
                                                        actual_metadata = metadata[0] if len(metadata[0]) > 0 else {}
                                                    else:
                                                        actual_metadata = metadata

                                                    # 处理实际metadata
                                                    if isinstance(actual_metadata, dict):
                                                        metadata_dict = actual_metadata
                                                        # 提取文件名作为标题，兼容不同的字段名
                                                        if 'name' in metadata_dict:
                                                            metadata_dict['filename'] = metadata_dict['name']  # 确保有filename字段
                                                        elif 'path' in metadata_dict:
                                                            # 从路径中提取文件名作为备用
                                                            import os
                                                            metadata_dict['filename'] = os.path.basename(metadata_dict['path'])
                                                        # 确保filename字段存在
                                                        if 'filename' not in metadata_dict:
                                                            metadata_dict['filename'] = '未知来源'
                                                    # 尝试安全转换为字典
                                                    elif isinstance(actual_metadata, (list, tuple)):
                                                        # 检查是否是键值对列表
                                                        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in actual_metadata):
                                                            metadata_dict = dict(actual_metadata)
                                                        else:
                                                            # 否则将整个列表作为一个值存储
                                                            metadata_dict = {'raw_metadata': str(actual_metadata)}
                                                    # 对于其他类型，转换为字符串存储
                                                    else:
                                                        metadata_dict = {'raw_metadata': str(actual_metadata)}
                                                except Exception as meta_error:
                                                    # 如果所有转换都失败，记录错误并使用空字典
                                                    self.logger.warning(f"metadata转换为字典失败: {str(meta_error)}")
                                                    metadata_dict = {'error': 'Failed to parse metadata', 'filename': '未知来源'}
                                            
                                            # 确保metadata_dict始终包含filename字段
                                            if 'filename' not in metadata_dict:
                                                metadata_dict['filename'] = '未知来源'

                                            # 添加到结果列表
                                            vector_results.append({
                                                'content': content_str,
                                                'metadata': metadata_dict,
                                                'score': similarity,
                                                'source': 'vector',
                                                'rank': i + 1,
                                                'distance': actual_distance  # 存储实际使用的距离值
                                            })
                                        except Exception as content_error:
                                            self.logger.warning(f"内容处理失败: {str(content_error)}")

                except Exception as e:
                    self.logger.error(f"AlayaLite搜索失败: {str(e)}")
                    # 尝试获取向量索引的实际维度信息
                    try:
                        vector_results.append({
                            'content': str(doc) if doc is not None else '',
                            'metadata': {'error': str(type_error)},
                            'score': similarity,
                            'source': 'vector',
                            'rank': i + 1,
                            'distance': actual_distance
                        })
                        collection = self.vdb_controller.get_collection(self.valves.VAULT_ID)
                        if collection and hasattr(collection, 'dim'):
                            self.logger.info(f"索引实际维度: {collection.dim}")
                    except:
                        pass
            else:
                # 如果不使用AlayaLite，记录警告信息
                self.logger.warning("AlayaLite未启用且ChromaDB已移除，无法执行向量检索")
                return []
            
            # 按相似度排序并返回top-k
            if self.valves.DEBUG:
                self.logger.info(f"通过AlayaLite检索到的有效结果数量: {len(vector_results)}")
            vector_results.sort(key=lambda x: x['score'], reverse=True)
            return vector_results[:self.valves.TOP_K_RETRIEVAL]
            
        except Exception as e:
            self.logger.error(f"向量检索失败: {str(e)}")
            return []
    
    async def _bm25_search(self, query: str) -> List[Dict]:
        """BM25关键词检索 - 基于词频和逆文档频率的精确匹配
        
        BM25是一种概率检索模型，特别擅长处理关键词匹配。
        它考虑词频(TF)、逆文档频率(IDF)和文档长度归一化。
        
        参数:
            query: 用户查询文本
            
        返回:
            List[Dict]: 按BM25分数排序的文档列表
            
        算法特点:
            - 词频饱和：避免高频词过度影响
            - 长度归一化：公平对待不同长度文档
            - 逆文档频率：突出稀有但重要的词
            - 精确匹配：对专业术语和实体名称效果好
        """
        try:
            if not self.bm25_index or not self.document_chunks:
                if self.valves.DEBUG:
                    self.logger.warning("BM25索引或文档块未初始化")
                return []
            
            # =================================================================
            # 查询预处理 - 智能分词和标准化
            # =================================================================
            query_tokens = self._tokenize_text(query)
            
            if not query_tokens:
                if self.valves.DEBUG:
                    self.logger.warning(f"查询分词结果为空: {query}")
                return []
            
            # =================================================================
            # BM25评分计算 - 计算每个文档的相关性分数
            # =================================================================
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # =================================================================
            # 动态阈值过滤 - 基于分数分布自适应过滤
            # =================================================================
            max_score = np.max(bm25_scores) if len(bm25_scores) > 0 else 0
            score_threshold = max(0.1, max_score * 0.1)  # 最高分的10%作为阈值，确保质量
            
            # 收集所有超过阈值的结果
            valid_results = []
            for idx, score in enumerate(bm25_scores):
                if score > score_threshold:
                    valid_results.append((idx, score))
            
            # =================================================================
            # 结果排序和格式化 - 按相关性分数降序排列
            # =================================================================
            valid_results.sort(key=lambda x: x[1], reverse=True)
            
            # 构建标准化结果格式
            bm25_results = []
            for i, (idx, score) in enumerate(valid_results[:self.valves.TOP_K_RETRIEVAL]):
                bm25_results.append({
                    'content': self.document_chunks[idx]['content'],
                    'metadata': self.document_chunks[idx]['metadata'],
                    'score': float(score),
                    'source': 'bm25',  # 标记检索来源
                    'rank': i + 1,     # 排名信息
                    'query_tokens': query_tokens  # 保存查询词用于后续分析
                })
            
            if self.valves.DEBUG and bm25_results:
                self.logger.info(f"BM25检索返回 {len(bm25_results)} 个结果，最高分: {bm25_results[0]['score']:.3f}")
            
            return bm25_results
            
        except Exception as e:
            self.logger.error(f"BM25检索失败: {str(e)}")
            return []
    
    def _merge_search_results(self, vector_results: List[Dict], bm25_results: List[Dict], query: str) -> List[Dict]:
        """合并向量检索和BM25检索结果"""
        try:
            # 归一化分数
            if vector_results:
                max_vector_score = max(r['score'] for r in vector_results)
                if max_vector_score > 0:
                    for result in vector_results:
                        result['normalized_score'] = result['score'] / max_vector_score
                else:
                    for result in vector_results:
                        result['normalized_score'] = 0
            
            if bm25_results:
                max_bm25_score = max(r['score'] for r in bm25_results)
                if max_bm25_score > 0:
                    for result in bm25_results:
                        result['normalized_score'] = result['score'] / max_bm25_score
                else:
                    for result in bm25_results:
                        result['normalized_score'] = 0
            
            # 合并结果并计算混合分数
            all_results = []
            
            # 添加向量检索结果
            for result in vector_results:
                result['hybrid_score'] = (
                    result['normalized_score'] * self.valves.VECTOR_WEIGHT
                )
                all_results.append(result)
            
            # 添加BM25检索结果
            for result in bm25_results:
                result['hybrid_score'] = (
                    result['normalized_score'] * self.valves.BM25_WEIGHT
                )
                all_results.append(result)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"合并检索结果失败: {str(e)}")
            return vector_results + bm25_results
    
    def _deduplicate_and_rank(self, results: List[Dict]) -> List[Dict]:
        """去重和排序结果"""
        try:
            # 确保results是列表类型
            if not isinstance(results, list):
                self.logger.warning("结果不是列表类型，已转换为空列表")
                return []
            
            # 基于内容去重
            seen_content = set()
            unique_results = []
            
            for result in results:
                try:
                    # 确保result是字典类型
                    if not isinstance(result, dict):
                        # 尝试将非字典类型的结果转换为字典
                        try:
                            result = {
                                'content': str(result) if result else '',
                                'score': 0.0
                            }
                        except:
                            self.logger.warning("无法将结果转换为有效格式，已跳过")
                            continue
                    
                    # 安全获取content并确保为字符串类型
                    content = result.get('content', '')
                    if not isinstance(content, str):
                        try:
                            content = str(content)
                        except:
                            content = ''
                            self.logger.warning("内容转换为字符串失败")
                    
                    # 计算内容哈希进行去重，避免unhashable type错误
                    try:
                        # 确保content的前200字符是可哈希的
                        content_segment = content[:200] if content else ''
                        content_hash = hash(content_segment)
                        
                        if content_hash not in seen_content:
                            seen_content.add(content_hash)
                            unique_results.append(result)
                    except TypeError as hash_error:
                        # 处理不可哈希的内容
                        self.logger.warning(f"内容哈希失败: {str(hash_error)}")
                        # 使用字符串表示作为备选去重方案
                        content_str = str(content_segment)
                        if content_str not in seen_content:
                            seen_content.add(content_str)
                            unique_results.append(result)
                except Exception as item_error:
                    self.logger.warning(f"处理单个结果失败: {str(item_error)}")
                    continue
            
            # 按混合分数排序，增加容错处理
            try:
                unique_results.sort(key=lambda x: x.get('hybrid_score', x.get('score', 0)), reverse=True)
            except Exception as sort_error:
                self.logger.warning(f"排序失败: {str(sort_error)}")
                # 如果排序失败，尝试使用备用排序键
                try:
                    unique_results.sort(key=lambda x: str(x.get('content', '')), reverse=False)
                except:
                    pass  # 如果备用排序也失败，就保持原样
            
            return unique_results
            
        except Exception as e:
            self.logger.error(f"去重和排序失败: {str(e)}")
            # 尝试返回原始结果的安全版本
            try:
                safe_results = []
                for result in results:
                    if isinstance(result, dict):
                        safe_results.append(result)
                return safe_results[:5]  # 只返回前5个结果作为安全保障
            except:
                return []
    
    async def _rerank_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """重排序模块 - 使用cross-encoder对检索结果进行重新排序"""
        try:
            if not documents:
                return documents
            
            if self.valves.DEBUG:
                self.logger.info(f"开始重排序 {len(documents)} 个文档")
            
            # 如果文档数量少于等于1，直接返回
            if len(documents) <= 1:
                return documents
            
            # 确保query是字符串类型
            if not isinstance(query, str):
                try:
                    query = str(query)
                except:
                    query = ""
                    self.logger.warning("查询转换为字符串失败，已重置为空字符串")
            
            # 准备输入对，确保所有内容都是字符串类型
            query_doc_pairs = []
            valid_docs = []  # 保存有效的文档，确保与query_doc_pairs长度匹配
            
            for doc in documents:
                try:
                    # 确保doc是字典类型
                    if not isinstance(doc, dict):
                        # 尝试将非字典类型的文档转换为字典
                        doc = {
                            'content': str(doc) if doc else '',
                            'score': 0.0
                        }
                    
                    # 安全获取content并确保为字符串类型
                    content = doc.get('content', '')
                    if not isinstance(content, str):
                        content = str(content)
                    
                    # 截断文档内容以避免超出模型限制
                    content = content[:512]  # 限制在512字符内
                    
                    # 确保query和content都是非空字符串
                    if query and content:
                        query_doc_pairs.append([query, content])
                        valid_docs.append(doc)
                except Exception as doc_error:
                    self.logger.warning(f"文档处理失败，已跳过: {str(doc_error)}")
                    continue
            
            # 如果没有有效的文档对，直接返回原始文档的前几个
            if not query_doc_pairs:
                self.logger.warning("没有有效的文档对用于重排序")
                return documents[:self.valves.TOP_K_RERANK]
            
            # 使用cross-encoder计算相关性分数
            try:
                if not hasattr(self, 'rerank_model') or self.rerank_model is None:
                    # 如果rerank模型未加载，使用简单的文本相似度
                    reranked_docs = self._simple_rerank(query, documents)
                else:
                    # 使用cross-encoder模型
                    scores = self.rerank_model.predict(query_doc_pairs)
                    
                    # 确保scores是有效的数值类型
                    if not isinstance(scores, (list, tuple, np.ndarray)):
                        raise ValueError("重排序模型返回的分数格式无效")
                    
                    # 为每个有效文档添加重排序分数
                    for i, doc in enumerate(valid_docs):
                        if i < len(scores):
                            try:
                                doc['rerank_score'] = float(scores[i])
                                doc['original_rank'] = i + 1
                            except:
                                doc['rerank_score'] = 0.0
                                self.logger.warning(f"分数转换失败，文档索引: {i}")
                    
                    # 按重排序分数排序
                    try:
                        reranked_docs = sorted(valid_docs, key=lambda x: x.get('rerank_score', 0), reverse=True)
                    except:
                        self.logger.warning("排序失败，返回原始文档列表")
                        reranked_docs = valid_docs
                    
                    if self.valves.DEBUG:
                        try:
                            top_scores = [doc['rerank_score'] for doc in reranked_docs[:3]]
                            self.logger.info(f"重排序完成，前3个文档的分数: {top_scores}")
                        except:
                            self.logger.info("重排序完成")
                
            except Exception as model_error:
                self.logger.warning(f"Cross-encoder重排序失败，使用简单重排序: {str(model_error)}")
                reranked_docs = self._simple_rerank(query, documents)
            
            # 返回top-k结果
            return reranked_docs[:self.valves.TOP_K_RERANK]
            
        except Exception as e:
            self.logger.error(f"重排序失败: {str(e)}")
            return documents[:self.valves.TOP_K_RERANK]
    
    def _simple_rerank(self, query: str, documents: List[Dict]) -> List[Dict]:
        """改进的重排序方法，基于多种相关性因子"""
        try:
            # 确保query是字符串类型
            if not isinstance(query, str):
                query = str(query)
            
            query_tokens = self._tokenize_text(query)
            query_tokens_set = set(query_tokens)
            
            for doc in documents:
                # 确保doc是字典类型
                if not isinstance(doc, dict):
                    continue
                
                # 安全获取content并确保为字符串类型
                content = doc.get('content', '')
                if not isinstance(content, str):
                    try:
                        content = str(content)
                    except:
                        content = ''
                        self.logger.warning(f"文档内容转换为字符串失败")
                
                # 分词处理
                content_tokens = []
                try:
                    content_tokens = self._tokenize_text(content)
                except Exception as token_error:
                    self.logger.warning(f"分词失败: {str(token_error)}")
                    # 简单分词作为备选方案
                    if hasattr(content, 'split'):
                        content_tokens = content.split()
                
                content_tokens_set = set(content_tokens)
                
                # 1. 关键词重叠度（Jaccard相似度）
                if len(query_tokens_set.union(content_tokens_set)) > 0:
                    jaccard_score = len(query_tokens_set.intersection(content_tokens_set)) / len(query_tokens_set.union(content_tokens_set))
                else:
                    jaccard_score = 0.0
                
                # 2. 查询覆盖度（查询词在文档中的覆盖比例）
                if len(query_tokens_set) > 0:
                    coverage_score = len(query_tokens_set.intersection(content_tokens_set)) / len(query_tokens_set)
                else:
                    coverage_score = 0.0
                
                # 3. 位置权重（查询词在文档开头的权重更高）
                position_score = 0.0
                try:
                    content_lower = content.lower()
                    query_lower = query.lower()
                    
                    # 检查查询词是否在文档前半部分出现
                    mid_point = len(content) // 2
                    if query_lower in content_lower[:mid_point]:
                        position_score = 0.8
                    elif query_lower in content_lower:
                        position_score = 0.4
                except Exception as pos_error:
                    self.logger.warning(f"位置权重计算失败: {str(pos_error)}")
                
                # 4. 文档长度惩罚（避免过长文档获得不当优势）
                length_penalty = 0.0
                try:
                    length_penalty = min(1.0, 500 / max(len(content), 1))
                except:
                    length_penalty = 1.0
                
                # 5. 源类型权重（向量检索和BM25的不同权重）
                source_weight = 1.0
                try:
                    if doc.get('source') == 'vector':
                        source_weight = 1.1  # 向量检索结果稍微加权
                    elif doc.get('source') == 'bm25':
                        source_weight = 1.0
                except:
                    source_weight = 1.0
                
                # 综合计算重排序分数
                semantic_score = 0.4 * jaccard_score + 0.3 * coverage_score + 0.2 * position_score + 0.1 * length_penalty
                
                # 结合原始检索分数
                original_score = 0.0
                try:
                    original_score = doc.get('hybrid_score', doc.get('score', 0))
                except:
                    original_score = 0.0
                
                # 确保分数是数字类型
                try:
                    original_score = float(original_score)
                except:
                    original_score = 0.0
                
                # 归一化原始分数
                normalized_original = min(1.0, max(0.0, original_score))
                
                # 最终分数：60%语义分数 + 40%原始分数，再乘以源权重
                try:
                    doc['rerank_score'] = (0.6 * semantic_score + 0.4 * normalized_original) * source_weight
                except:
                    doc['rerank_score'] = 0.0
                
                # 添加调试信息
                if self.valves.DEBUG:
                    try:
                        doc['debug_scores'] = {
                            'jaccard': jaccard_score,
                            'coverage': coverage_score,
                            'position': position_score,
                            'length_penalty': length_penalty,
                            'semantic': semantic_score,
                            'original': normalized_original,
                            'source_weight': source_weight
                        }
                    except:
                        pass
            
            # 按重排序分数排序，增加容错处理
            try:
                return sorted(documents, key=lambda x: x.get('rerank_score', 0), reverse=True)
            except:
                self.logger.warning("排序失败，返回原始文档列表")
                return documents
            
        except Exception as e:
            self.logger.error(f"简单重排序失败: {str(e)}")
            return documents
    
    def _filter_relevant_documents(self, query: str, documents: List[Dict]) -> List[Dict]:
        """过滤相关性较低的文档"""
        try:
            if not documents:
                return documents
            
            # 设置相关性阈值
            relevance_threshold = 0.1  # 最低相关性分数
            
            # 过滤低分文档
            filtered_docs = []
            for doc in documents:
                score = doc.get('rerank_score', doc.get('hybrid_score', doc.get('score', 0)))
                if score >= relevance_threshold:
                    filtered_docs.append(doc)
            
            # 如果过滤后文档太少，保留前3个最高分文档
            if len(filtered_docs) < 2 and len(documents) >= 2:
                filtered_docs = sorted(documents, 
                                     key=lambda x: x.get('rerank_score', x.get('hybrid_score', x.get('score', 0))), 
                                     reverse=True)[:3]
            
            if self.valves.DEBUG:
                self.logger.info(f"文档过滤：{len(documents)} -> {len(filtered_docs)}")
            
            return filtered_docs
            
        except Exception as e:
            if self.valves.DEBUG:
                self.logger.warning(f"文档过滤失败: {e}")
            return documents
    
    async def _generate_answer(self, query: str, documents: List[Dict], messages: List[dict]) -> str:
        """生成最终答案 - RAG系统的答案生成阶段
        
        这是RAG流程的最后一步，将检索到的相关文档与用户查询结合，
        通过大语言模型生成准确、相关且有引用的答案。
        
        参数:
            query: 用户原始查询
            documents: 经过检索和重排序的相关文档列表
            messages: 对话历史（用于上下文理解）
            
        返回:
            str: 生成的答案文本，包含引用信息
            
        处理流程:
            1. 文档相关性最终过滤
            2. 上下文构建和长度控制
            3. 提示词工程
            4. LLM调用生成答案
            5. 答案后处理和引用添加
        """
        try:
            # =================================================================
            # 输入验证 - 确保有可用的检索结果
            # =================================================================
            # 确保documents是列表类型
            if not isinstance(documents, list):
                documents = []
                self.logger.warning("文档列表类型错误，已重置为空列表")
            
            if not documents:
                return "抱歉，我没有找到相关的信息来回答您的问题。请尝试使用不同的关键词或更具体的描述。"
            
            # =================================================================
            # 文档数据清洗 - 确保每个文档都是字典类型并包含必要字段
            # =================================================================
            clean_documents = []
            for doc in documents:
                if isinstance(doc, dict):
                    # 确保content字段存在且是字符串类型
                    if 'content' not in doc or not isinstance(doc['content'], str):
                        try:
                            doc['content'] = str(doc.get('content', ''))
                        except:
                            doc['content'] = ''
                    
                    # 确保metadata字段存在且是字典类型
                    if 'metadata' not in doc or not isinstance(doc['metadata'], dict):
                        doc['metadata'] = dict(doc.get('metadata', {})) if doc.get('metadata') else {}
                    
                    clean_documents.append(doc)
                else:
                    # 尝试将非字典类型的文档转换为字典
                    try:
                        clean_doc = {
                            'content': str(doc) if doc else '',
                            'metadata': {},
                            'score': 0.0
                        }
                        clean_documents.append(clean_doc)
                    except:
                        self.logger.warning("无法将文档转换为有效格式，已跳过")
                        continue
            
            # =================================================================
            # 文档质量过滤 - 最后一轮相关性检查
            # =================================================================
            filtered_documents = self._filter_relevant_documents(query, clean_documents)
            
            if not filtered_documents:
                return "抱歉，检索到的文档与您的问题相关性较低，无法提供准确的答案。请尝试重新表述您的问题。"
            
            # =================================================================
            # 降级模式 - 无LLM时提供文档摘要
            # =================================================================
            if not self.openai_client:
                if self.valves.DEBUG:
                    self.logger.info("OpenAI客户端不可用，使用文档摘要模式")
                    
                # 构建结构化文档摘要
                doc_summaries = []
                sources = []
                for i, doc in enumerate(filtered_documents[:3], 1):
                    content = doc['content'][:400] + "..." if len(doc['content']) > 400 else doc['content']
                    source = doc['metadata'].get('filename', '未知来源')
                    score = doc.get('rerank_score', doc.get('hybrid_score', doc.get('score', 0)))
                    doc_summaries.append(f"{i}. 来源：{source}\n相关性：{score:.2f}\n内容：{content}")
                    if source not in sources:
                        sources.append(source)
                
                summary = f"基于检索到的相关文档，以下是相关信息：\n\n" + "\n\n".join(doc_summaries)
                
                if sources:
                    summary += f"\n\n**信息来源：**\n"
                    for source in sources[:3]:
                        summary += f"- {source}\n"
                
                return summary
            
            # =================================================================
            # 上下文构建 - 智能选择和组织相关文档
            # =================================================================
            context_parts = []
            sources = []
            total_context_length = 0
            max_context_length = 3000  # 限制总上下文长度，避免超出模型限制
            
            for i, doc in enumerate(filtered_documents[:self.valves.TOP_K_FINAL], 1):
                content = doc['content']
                source = doc['metadata'].get('filename', '未知来源')
                score = doc.get('rerank_score', doc.get('hybrid_score', doc.get('score', 0)))
                
                # 动态调整文档长度
                remaining_length = max_context_length - total_context_length
                if remaining_length <= 0:
                    break
                
                # 根据相关性分数调整文档长度
                if score > 0.7:
                    doc_length = min(800, remaining_length)
                elif score > 0.4:
                    doc_length = min(600, remaining_length)
                else:
                    doc_length = min(400, remaining_length)
                
                truncated_content = content[:doc_length]
                if len(content) > doc_length:
                    truncated_content += "..."
                
                context_parts.append(f"文档{i}（来源：{source}，相关性：{score:.2f}）：\n{truncated_content}")
                total_context_length += len(truncated_content)
                
                if source not in sources:
                    sources.append(source)
            
            context = "\n\n".join(context_parts)
            
            # 构建对话历史
            conversation_history = ""
            if messages and len(messages) > 1:
                recent_messages = messages[-2:]  # 获取最近2条消息
                for msg in recent_messages:
                    if msg.get('role') == 'user':
                        conversation_history += f"User: {msg.get('content', '')[:150]}\n"
                    elif msg.get('role') == 'assistant':
                        conversation_history += f"Assistant: {msg.get('content', '')[:150]}\n"
            
            # Build system prompt for Wikipedia knowledge Q&A
            system_prompt = """You are a professional knowledge Q&A AI assistant specialized in answering knowledge questions based on Wikipedia content. Follow these rules:

1. **Document-based answers**: Answer questions strictly based on the provided Wikipedia document content to ensure accuracy and reliability
2. **Knowledge-focused responses**: Provide comprehensive and accurate knowledge information including definitions, background, and important facts
3. **Structured presentation**: Use clear logical structure to organize answers suitable for knowledge learning and understanding
4. **Encyclopedia style**: Use objective, neutral encyclopedic expression, avoiding subjective judgments
5. **Contextual connections**: When appropriate, provide relevant background information and knowledge connections
6. **English expression**: Use accurate and fluent English for responses

Answer strategies:
- **Person questions**: Include biographical overview, major achievements, historical significance
- **Concept questions**: Provide clear definitions, core characteristics, important applications
- **Historical events**: Explain background, process, impact and significance
- **Geographic questions**: Describe location, features, importance
- **Science & technology**: Explain principles, development history, application value

Answer format:
- Directly answer the core question
- Provide key facts and detailed information
- Use bullet points or paragraphs to organize complex content
- Maintain objective and neutral expression style"""
            
            # Build user prompt
            user_prompt = f"""Conversation history:
{conversation_history}

Relevant documents:
{context}

User question: {query}

Please answer the question accurately based on the above documents. If there is no directly relevant information in the documents, please clearly state this and suggest how the user can obtain the needed information."""
            
            # 调用LLM生成答案
            response = await self.openai_client.chat.completions.create(
                model=self.valves.MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.05,  # 更低的温度以获得更一致和准确的答案
                max_tokens=1200
            )
            
            answer = response.choices[0].message.content.strip()
            
            # 检查答案质量
            if len(answer) < 20:
                answer = "基于提供的文档，我无法生成完整的答案。请尝试更具体的问题或检查文档内容的相关性。"
            
            # 添加文档来源信息
            if sources and "信息来源" not in answer:
                answer += "\n\n📚 **References:**\n"
                for source in sources[:3]:  # Show up to 3 sources
                    answer += f"- {source}\n"
            
            return answer
            
        except Exception as e:
            self.logger.error(f"Answer generation failed: {str(e)}")
            return f"Sorry, an error occurred while generating the answer. Please try again later or contact technical support. Error: {str(e)}"
    
    def evaluate_query_expansion(self, original_query: str, expanded_queries: List[str], 
                               retrieved_docs: List[Dict]) -> Dict[str, Any]:
        """评估查询扩展效果
        
        Args:
            original_query: 原始查询
            expanded_queries: 扩展后的查询列表
            retrieved_docs: 检索到的文档列表
            
        Returns:
            评估结果字典
        """
        try:
            evaluation_result = {
                'original_query': original_query,
                'expanded_queries': expanded_queries,
                'expansion_metrics': {},
                'retrieval_metrics': {},
                'overall_score': 0.0
            }
            
            # 1. 查询扩展质量评估
            expansion_metrics = {
                'num_expanded_queries': len(expanded_queries),
                'avg_query_length': sum(len(q.split()) for q in expanded_queries) / len(expanded_queries) if expanded_queries else 0,
                'diversity_score': self._calculate_query_diversity(expanded_queries),
                'english_ratio': self._calculate_english_ratio(expanded_queries)
            }
            
            # 2. 检索效果评估
            retrieval_metrics = {
                'num_retrieved_docs': len(retrieved_docs),
                'avg_relevance_score': sum(doc.get('score', 0) for doc in retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0,
                'source_diversity': len(set(doc.get('source', '') for doc in retrieved_docs)),
                'content_coverage': self._calculate_content_coverage(retrieved_docs)
            }
            
            # 3. 计算综合评分
            overall_score = self._calculate_overall_expansion_score(expansion_metrics, retrieval_metrics)
            
            evaluation_result['expansion_metrics'] = expansion_metrics
            evaluation_result['retrieval_metrics'] = retrieval_metrics
            evaluation_result['overall_score'] = overall_score
            
            if self.valves.DEBUG:
                self.logger.info(f"Query expansion evaluation: {evaluation_result}")
            
            return evaluation_result
            
        except Exception as e:
            self.logger.error(f"Query expansion evaluation failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_query_diversity(self, queries: List[str]) -> float:
        """计算查询多样性分数"""
        if len(queries) <= 1:
            return 0.0
        
        # 计算查询间的词汇重叠度
        query_tokens = [set(q.lower().split()) for q in queries]
        total_pairs = len(queries) * (len(queries) - 1) / 2
        overlap_sum = 0
        
        for i in range(len(query_tokens)):
            for j in range(i + 1, len(query_tokens)):
                intersection = len(query_tokens[i] & query_tokens[j])
                union = len(query_tokens[i] | query_tokens[j])
                overlap = intersection / union if union > 0 else 0
                overlap_sum += overlap
        
        avg_overlap = overlap_sum / total_pairs if total_pairs > 0 else 0
        diversity_score = 1 - avg_overlap  # 重叠度越低，多样性越高
        
        return max(0.0, min(1.0, diversity_score))
    
    def _calculate_english_ratio(self, queries: List[str]) -> float:
        """计算英文查询比例"""
        if not queries:
            return 0.0
        
        english_queries = 0
        for query in queries:
            # 简单的英文检测：如果查询中英文字符占比超过80%则认为是英文查询
            english_chars = sum(1 for c in query if c.isascii() and c.isalpha())
            total_chars = sum(1 for c in query if c.isalpha())
            
            if total_chars > 0 and english_chars / total_chars > 0.8:
                english_queries += 1
        
        return english_queries / len(queries)
    
    def _calculate_content_coverage(self, docs: List[Dict]) -> float:
        """计算内容覆盖度"""
        if not docs:
            return 0.0
        
        # 计算文档内容的总长度和平均长度
        total_length = sum(len(doc.get('content', '')) for doc in docs)
        avg_length = total_length / len(docs)
        
        # 归一化到0-1范围，假设平均1000字符为满分
        coverage_score = min(1.0, avg_length / 1000)
        
        return coverage_score
    
    def _calculate_overall_expansion_score(self, expansion_metrics: Dict, retrieval_metrics: Dict) -> float:
        """计算查询扩展综合评分"""
        try:
            # 权重设置
            weights = {
                'diversity': 0.25,
                'english_ratio': 0.20,
                'retrieval_quality': 0.30,
                'content_coverage': 0.25
            }
            
            # 各项分数
            diversity_score = expansion_metrics.get('diversity_score', 0)
            english_score = expansion_metrics.get('english_ratio', 0)
            retrieval_score = min(1.0, retrieval_metrics.get('avg_relevance_score', 0) / 0.8)  # 假设0.8为满分
            coverage_score = retrieval_metrics.get('content_coverage', 0)
            
            # 加权计算
            overall_score = (
                diversity_score * weights['diversity'] +
                english_score * weights['english_ratio'] +
                retrieval_score * weights['retrieval_quality'] +
                coverage_score * weights['content_coverage']
            )
            
            return round(overall_score, 3)
            
        except Exception as e:
            self.logger.error(f"Overall score calculation failed: {str(e)}")
            return 0.0

    # =============================================================================
    # 文件监控相关方法
    # =============================================================================
    async def _start_file_monitoring(self):
        """启动DATA_PATH目录监控，实时更新BM25索引"""
        try:
            # 直接使用DATA_PATH作为监控路径
            monitor_path = self.valves.DATA_PATH
            
            if not os.path.exists(monitor_path):
                self.logger.warning(f"监控路径不存在: {monitor_path}")
                # 创建监控目录
                os.makedirs(monitor_path, exist_ok=True)
                self.logger.info(f"已创建监控目录: {monitor_path}")
                
            # 初始化文件状态记录
            self._file_status = self._get_current_file_status(monitor_path)
            self._last_monitor_time = time.time() if time else 0
            self._stop_event = threading.Event() if threading else None
            
            # 根据配置选择监控方式
            if self.valves.USE_WATCHDOG and watchdog and VaultFileHandler:
                # 使用watchdog进行实时监控
                self._observer = watchdog.observers.Observer()
                event_handler = VaultFileHandler(self)
                self._observer.schedule(event_handler, monitor_path, recursive=True)
                self._observer.start()
                if self.valves.DEBUG:
                    self.logger.info(f"使用watchdog启动实时文件监控: {monitor_path}")
            elif self.valves.USE_WATCHDOG and (not watchdog or not VaultFileHandler):
                self.logger.warning("watchdog监控不可用，尝试使用轮询方式")
            elif threading:
                # 使用轮询方式监控
                self._monitor_thread = threading.Thread(
                    target=self._poll_files,
                    args=(monitor_path,),
                    daemon=True
                )
                self._monitor_thread.start()
                if self.valves.DEBUG:
                    self.logger.info(f"使用轮询方式启动文件监控: {monitor_path}，间隔 {self.valves.MONITOR_INTERVAL} 秒")
            else:
                self.logger.warning("无法启动文件监控，缺少必要的依赖库")
                
        except Exception as e:
            self.logger.error(f"启动文件监控失败: {str(e)}")
            
    async def _stop_file_monitoring(self):
        """停止文件监控"""
        try:
            if self._stop_event:
                self._stop_event.set()
                
            if self._observer and self.valves.USE_WATCHDOG:
                self._observer.stop()
                self._observer.join(timeout=5)
                self._observer = None
                
            if self._monitor_thread and not self.valves.USE_WATCHDOG:
                self._monitor_thread.join(timeout=5)
                self._monitor_thread = None
                
            if self.valves.DEBUG:
                self.logger.info("文件监控已停止")
                
        except Exception as e:
            self.logger.error(f"停止文件监控失败: {str(e)}")
            
    def _poll_files(self, directory):
        """轮询方式监控文件变化"""
        try:
            while self._stop_event and not self._stop_event.is_set():
                current_time = time.time() if time else 0
                # 检查是否达到监控间隔
                if current_time - self._last_monitor_time >= self.valves.MONITOR_INTERVAL:
                    self._last_monitor_time = current_time
                    
                    # 异步调用检查文件变化
                    if asyncio:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(self._check_file_changes(directory))
                        loop.close()
                    else:
                        self.logger.warning("asyncio不可用，无法检查文件变化")
                    
                # 短暂休眠避免CPU占用过高
                if time:
                    time.sleep(1)
                
        except Exception as e:
            self.logger.error(f"文件轮询监控失败: {str(e)}")
            
    async def _check_file_changes(self, directory):
        """检查目录中的文件变化"""
        try:
            # 获取当前文件状态
            current_status = self._get_current_file_status(directory)
            
            # 检查新增、修改或删除的文件
            has_changes = False
            
            # 检查新增或修改的文件
            for file_path, status in current_status.items():
                if file_path not in self._file_status or self._file_status[file_path] != status:
                    has_changes = True
                    break
                    
            # 检查删除的文件
            for file_path in self._file_status:
                if file_path not in current_status:
                    has_changes = True
                    break
                    
            # 如果有文件变化，更新BM25索引
            if has_changes:
                self._file_status = current_status
                await self._update_bm25_index()
                
        except Exception as e:
            self.logger.error(f"检查文件变化失败: {str(e)}")
            
    async def _update_bm25_index(self):
        """更新BM25索引"""
        try:
            if self.valves.DEBUG:
                self.logger.info("开始更新BM25索引...")
                
            # 重新加载文档并分块
            documents = await self._load_documents_for_indexing()
            if documents:
                self.document_chunks = self._chunk_documents_for_indexing(documents)
                # 重新构建BM25索引
                self._build_bm25_index()
                
                if self.valves.DEBUG:
                    self.logger.info(f"BM25索引更新完成，包含 {len(self.document_chunks)} 个文档块")
            else:
                self.logger.warning("没有找到文档，BM25索引未更新")
                
        except Exception as e:
            self.logger.error(f"更新BM25索引失败: {str(e)}")
            
    def _get_current_file_status(self, directory):
        """获取目录中文件的当前状态（修改时间和大小）"""
        file_status = {}
        try:
            supported_extensions = {'.txt', '.md', '.json'}
            
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    if file_ext in supported_extensions:
                        try:
                            # 获取文件修改时间和大小
                            stat = os.stat(file_path)
                            file_status[file_path] = (stat.st_mtime, stat.st_size)
                        except Exception as e:
                            self.logger.warning(f"获取文件状态失败 {file_path}: {str(e)}")
                            
        except Exception as e:
            self.logger.error(f"获取文件状态失败: {str(e)}")
            
        return file_status


# =============================================================================
# 文件监控事件处理器（条件导入，避免依赖缺失时的错误）
# =============================================================================
VaultFileHandler = None
if watchdog and hasattr(watchdog.events, 'FileSystemEventHandler'):
    class VaultFileHandler(watchdog.events.FileSystemEventHandler):
        """watchdog文件监控事件处理器"""
        
        def __init__(self, pipeline):
            self.pipeline = pipeline
            
        def on_created(self, event):
            """处理文件创建事件"""
            if not event.is_directory:
                self._handle_file_change(event.src_path)
                
        def on_modified(self, event):
            """处理文件修改事件"""
            if not event.is_directory:
                self._handle_file_change(event.src_path)
                
        def on_deleted(self, event):
            """处理文件删除事件"""
            if not event.is_directory:
                self._handle_file_change(event.src_path)
                
        def on_moved(self, event):
            """处理文件移动事件"""
            if not event.is_directory:
                self._handle_file_change(event.dest_path)
                
        def _handle_file_change(self, file_path):
            """处理文件变化"""
            # 检查文件扩展名
            supported_extensions = {'.txt', '.md', '.json'}
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in supported_extensions:
                if self.pipeline.valves.DEBUG:
                    self.pipeline.logger.info(f"检测到文件变化: {file_path}")
                    
                # 异步更新BM25索引
                if asyncio:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.pipeline._update_bm25_index())
                    loop.close()