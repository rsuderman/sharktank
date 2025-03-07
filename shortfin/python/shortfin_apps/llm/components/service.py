# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import logging
import os
from pathlib import Path


import shortfin as sf
import shortfin.array as sfnp

from .batcher import LlmBatcherProcess
from ...utils import ServiceBase, prog_isolations

from .kvcache.base_attention_cache import BasePagedAttentionCache
from .kvcache.trie_attention_cache import TriePagedAttentionCache

from .kvcache.page_pool import PagePoolConfig, PagePool 
from .config_struct import ModelParams, ServerParams
from .manager import SystemManager
from .messages import LlmInferenceExecRequest, InferencePhase
from .tokenizer import Tokenizer
from .service_debug_dumper import SERVICE_DEBUG_DUMPER

logger = logging.getLogger(__name__)


class GenerateService(ServiceBase):
    """Top level service interface for generating text against a model."""

    inference_program: sf.Program
    prefill_functions: dict[int, sf.ProgramFunction]
    decode_functions: dict[int, sf.ProgramFunction]

    def __init__(
        self,
        *,
        name: str,
        sysman: SystemManager,
        tokenizer: Tokenizer,
        model_params: ModelParams,
        server_params: "ServerParams",
        program_isolation: str = "per_call",
    ):
        super().__init__(sysman)
        self.name = name

        # Application objects.
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.server_params = server_params
        self.main_worker = sysman.ls.create_worker(f"{name}-inference")
        self.main_fiber = sysman.ls.create_fiber(self.main_worker)

        page_pool_config = PagePoolConfig(
            dtype=model_params.attn_dtype,
            alloc_page_count=model_params.paged_kv_cache.device_block_count,
            paged_kv_block_size_elements=model_params.paged_kv_block_size_elements,
        )
        page_pool = PagePool(
            devices=self.main_fiber.devices_dict.values(), config=page_pool_config
        )
        if server_params.prefix_sharing_algorithm == "trie":
            self.page_cache = TriePagedAttentionCache(
                page_pool=page_pool,
                tokens_per_page=model_params.paged_kv_cache.block_seq_stride,
            )
        elif server_params.prefix_sharing_algorithm == "none":
            self.page_cache = BasePagedAttentionCache(
                page_pool=page_pool,
                tokens_per_page=model_params.paged_kv_cache.block_seq_stride,
            )
        else:
            raise ValueError(
                f"Unknown prefix_sharing_algorithm {server_params.prefix_sharing_algorithm}. Currently only supporting 'trie' and 'none'."
            )

        self.program_isolation = prog_isolations[program_isolation]

    def start(self):
        self.inference_program = sf.Program(
            modules=[
                sf.ProgramModule.parameter_provider(
                    self.sysman.ls, *self.inference_parameters["main"]
                )
            ]
            + self.inference_modules["main"],
            devices=self.sysman.ls.devices,
            trace_execution=False,
            isolation=self.program_isolation,
        )
        # Resolve prefill entrypoints.
        self.prefill_functions = {}
        for bs in self.model_params.prefill_batch_sizes:
            self.prefill_functions[bs] = self.inference_program[
                f"{self.model_params.module_name}.prefill_bs{bs}"
            ]
        # Resolve decode entrypoints.
        self.decode_functions = {}
        for bs in self.model_params.decode_batch_sizes:
            self.decode_functions[bs] = self.inference_program[
                f"{self.model_params.module_name}.decode_bs{bs}"
            ]

        # Scope dependent objects.
        self.batcher = LlmBatcherProcess(
            cache=self.page_cache,
            fiber=self.main_fiber,
            model_params=self.model_params,
            decode_functions=self.decode_functions,
            prefill_functions=self.prefill_functions)

        # Start persistent processes.
        self.batcher.launch()

    def shutdown(self):
        self.batcher.shutdown()

    def __repr__(self):
        return (
            f"ServiceManager(\n"
            f"  model_params={self.model_params}\n"
            f"  inference_modules={self.inference_modules}\n"
            f"  page_cache={self.page_cache}\n"
            f")"
        )
