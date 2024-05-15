# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import asyncio
import argparse
import numpy
import sys

from transformers import LlamaTokenizer  # type: ignore

from iree.runtime import (  # type: ignore
    HalElementType,
)

from shortfin.framework.session import DeviceSession

from shortfin.llm.attn_block_cache import (
    create_attn_block_cache_module,
    AttnBlockCache,
)

from shortfin.llm.config import (
    CacheParams,
    ModelParams,
    ServiceParams,
)

from shortfin.llm.impl.service_v1 import GenerateServiceV1
from shortfin.llm.service import GenerateRequest


def setup(vmfb_path, config_path, gguf_path):
    from iree.runtime._binding import disable_leak_checker  # type: ignore

    model_params = ModelParams.load_json(config_path)

    cache_params = CacheParams(
        model=model_params, device_block_count=128, block_pos_stride=16
    )

    disable_leak_checker()
    session = DeviceSession(uri="local-task", queue_count=2)
    attn_block_cache = AttnBlockCache(session, cache_params)

    lms = session.create_module_set(model_params.module_name, context_count=1)
    lms.load_io_module(gguf_path)
    lms.load_vmfb(vmfb_path)
    lms.add(create_attn_block_cache_module(attn_block_cache))
    lms.initialize()

    params = ServiceParams(cache=cache_params, model=model_params)
    service = GenerateServiceV1(session=session, params=params, cache=attn_block_cache)
    return service


def map_buffer(value):
    mapped = value.map()
    return mapped.asarray(value.shape, HalElementType.map_to_dtype(value.element_type))


async def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", help="name of hugginface tokenizer to use")
    parser.add_argument("--config", help="json config file with hyperparameters")
    parser.add_argument("--vmfb", help="vmfb with compiler LLM kernels")
    parser.add_argument("--gguf", help="gguf file containing modle coefficients")
    parsed = parser.parse_args(argv)

    hf_path = "openlm-research/open_llama_3b_v2" # parsed.tokenizer
    config_path = "/tmp/batch_llama_v1.json" # parsed.config
    vmfb_path = "/tmp/batch_llama_v1.vmfb" # parsed.vmfb
    gguf_path = "/media/rsuderman/Disk2/Models/llama.gguf" # parsed.gguf

    service = setup(vmfb_path, config_path, gguf_path)
    tokenizer = LlamaTokenizer.from_pretrained(hf_path)
    state = service.start()

    for line in ["one two three four"]:
        prompt = line.strip()
        if not prompt:
            break

        input_ids = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()
        print(input_ids)
        request = GenerateRequest("request_id", prompt, input_ids)
        await state.set_sequences([request])
        logits = await state.prefill()

        mapped_logits = map_buffer(logits.value)
        top_tokens = numpy.argmax(mapped_logits, axis=-1)[:, -1].tolist()
        print(top_tokens)
        print(tokenizer.decode(top_tokens))

        await state.set_decode_step(top_tokens[:1])
        result = await state.decode()
        mapped_result = map_buffer(result.value)
        top_tokens = numpy.argmax(mapped_result, axis=-1)
        top_tokens = top_tokens[:, 0].tolist()
        print(top_tokens)
        print(tokenizer.decode(top_tokens))

        await state.recycle()

    service.shutdown()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
