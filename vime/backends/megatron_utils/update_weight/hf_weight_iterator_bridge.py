import dataclasses

from vime.utils import megatron_bridge_utils
from vime.utils.misc import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


def _patch_bridge_expert_cache_to_cpu():
    """Patch GPT-OSS Bridge to cache merged expert weights on CPU."""
    try:
        from megatron.bridge.models.gpt_oss.gpt_oss_bridge import GPTOSSBridge
    except ImportError:
        return

    if getattr(GPTOSSBridge, "_cpu_cache_patched", False):
        return

    original = GPTOSSBridge.maybe_modify_converted_hf_weight

    def patched(self, task, converted_weights_dict):
        cpu_dict = {key: value.cpu() for key, value in converted_weights_dict.items()}
        result = original(self, task, cpu_dict)
        return {key: value.cuda() for key, value in result.items()} if result else result

    GPTOSSBridge.maybe_modify_converted_hf_weight = patched
    GPTOSSBridge._cpu_cache_patched = True


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        import vime_plugins.megatron_bridge  # noqa: F401

        self._bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(
            AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
        )
        _patch_bridge_expert_cache_to_cpu()

    def get_hf_weight_chunks(self, megatron_local_weights, progress_desc: str = "Update weights"):
        del progress_desc
        renamed_local_weights = {
            strip_param_name_prefix(name): weight for name, weight in megatron_local_weights.items()
        }
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_local_weights)
            named_weights = self._bridge.export_hf_weights(
                self.model,
                cpu=False,
                conversion_tasks=conversion_tasks,
            )

            def streaming_quantized():
                for hf_param_name, weight, megatron_param_name in named_weights:
                    processed_weight = postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=megatron_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    )
                    yield from quantize_params(
                        args=self.args,
                        megatron_name=megatron_param_name,
                        converted_named_params=[(hf_param_name, processed_weight)],
                        quantization_config=self.quantization_config,
                    )

            yield from chunk_named_params_by_size(
                streaming_quantized(),
                chunk_size=self.args.update_weight_buffer_size,
            )


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    def handle_one(task):
        if task is None or task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert weight_dict_key in new_weight_dict, (
            f"{weight_dict_key=} not in new_weight_dict "
            f"({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"
        )
        return dataclasses.replace(task, param_weight=new_weight_dict[weight_dict_key].cuda())

    return _MapWithLen(handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, values):
        self.fn = fn
        self.values = values

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        for value in self.values:
            yield self.fn(value)
