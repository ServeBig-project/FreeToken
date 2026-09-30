"""CPU-only independent numerical acceptance of the public DFlash model interface."""

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import padded_attention
import reference
from fixture import checkpoint, quantized


TOLERANCES = {"float32": {"max_abs": 2e-5, "nrmse": 2e-6},
              "bfloat16": {"max_abs": 0.0625, "nrmse": 0.01}}
MODES = [(True, 4095)] * 5 + [(False, -1)]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def array(tensor):
    return tensor.detach().float().numpy().astype(np.float64)


def metrics(actual, expected):
    require(actual.shape == expected.shape, f"Shape differs: {actual.shape} versus {expected.shape}")
    require(np.isfinite(actual).all() and np.isfinite(expected).all(), "Nonfinite numeric result")
    difference = actual - expected
    rmse = float(np.sqrt(np.mean(difference * difference)))
    return {"max_abs": float(np.max(np.abs(difference))), "rmse": rmse,
            "nrmse": rmse / max(float(np.sqrt(np.mean(expected * expected))), 1e-12)}


def append_history(history, request_ids, positions, keys, values):
    for request_id in np.unique(request_ids):
        selected = np.flatnonzero(request_ids == request_id)
        new = (torch.from_numpy(positions[selected].copy()), keys[selected].clone(), values[selected].clone())
        if request_id in history:
            new = tuple(torch.cat((old, added)) for old, added in zip(history[request_id], new))
        history[request_id] = new


def append_reference(history, request_ids, positions, keys, values):
    for request_id in np.unique(request_ids):
        selected = request_ids == request_id
        new = (positions[selected].copy(), keys[selected].copy(), values[selected].copy())
        if request_id in history:
            new = tuple(np.concatenate((old, added)) for old, added in zip(history[request_id], new))
        history[request_id] = new


def scenario(model, config, parameters, dtype, lengths, block_lengths, extensions, report):
    label = str(dtype).removeprefix("torch.")
    rounding = reference.bf16 if dtype == torch.bfloat16 else reference.ideal
    parameters = quantized(parameters, rounding)
    random = np.random.default_rng(7331 + len(lengths))
    histories = [{} for _ in MODES]
    rounded_histories = [{} for _ in MODES]
    ideal_histories = [{} for _ in MODES]
    cursors = np.zeros(len(lengths), dtype=np.int64)
    embedding = torch.tensor(random.standard_normal((47, 32)), dtype=dtype)
    embedding_copy = embedding.clone()
    head = random.standard_normal((47, 32)) / np.sqrt(32)
    alternate_head = random.standard_normal((47, 32)) / np.sqrt(32)

    def compare(name, actual, expected, enforce=True):
        measured = metrics(actual, expected)
        report["errors"].append({"name": name, "enforced": enforce, **measured})
        if enforce:
            for key, limit in TOLERANCES[label].items():
                require(measured[key] <= limit, f"{name}: {key}={measured[key]:.9g} exceeds {limit}")

    def context(counts, round_index):
        request_ids = np.repeat(np.arange(len(counts)), counts)
        positions = np.concatenate([np.arange(cursors[index], cursors[index] + count) for index, count in enumerate(counts)])
        permutation = random.permutation(len(positions))
        request_ids, positions = request_ids[permutation], positions[permutation]
        features = torch.tensor(random.standard_normal((len(positions), 256)), dtype=dtype)
        features_copy = features.clone()
        position_tensor = torch.from_numpy(positions.copy())
        expected = reference.project_context(array(features), positions, parameters, config, rounding)
        ideal = reference.project_context(array(features), positions, parameters, config)
        seen = []

        def store(index, keys, values):
            seen.append(index)
            require(tuple(keys.shape) == (len(positions), 2, 8), "Context key shape is incorrect")
            require(tuple(values.shape) == (len(positions), 2, 8), "Context value shape is incorrect")
            require(keys.dtype == values.dtype == dtype, "Context dtype is incorrect")
            compare(f"round{round_index}/context/layer{index}/k", array(keys), expected[index][0])
            compare(f"round{round_index}/context/layer{index}/v", array(values), expected[index][1])
            append_history(histories[index], request_ids, positions, keys, values)
            append_reference(rounded_histories[index], request_ids, positions, *expected[index])
            append_reference(ideal_histories[index], request_ids, positions, *ideal[index])

        model.project_context(features, position_tensor, store)
        require(sorted(seen) == list(range(6)), f"Context callbacks missing or duplicated: {seen}")
        require(torch.equal(features, features_copy), "project_context changed features")
        require(np.array_equal(position_tensor.numpy(), positions), "project_context changed positions")
        cursors[:] += counts

    def block(round_index):
        request_ids = np.repeat(np.arange(len(block_lengths)), block_lengths)
        positions = np.concatenate([np.arange(cursors[index], cursors[index] + count)
                                    for index, count in enumerate(block_lengths)])
        token_ids = np.concatenate([np.array([index + 1] + [46] * (count - 1))
                                    for index, count in enumerate(block_lengths)])
        permutation = random.permutation(len(positions))
        request_ids, positions, token_ids = request_ids[permutation], positions[permutation], token_ids[permutation]
        noise = embedding[token_ids].clone()
        noise_copy = noise.clone()
        position_tensor = torch.from_numpy(positions.copy())
        expected, trace = reference.forward(array(noise), positions, request_ids, rounded_histories,
                                            parameters, config, MODES, rounding)
        ideal, _ = reference.forward(array(noise), positions, request_ids, ideal_histories,
                                     parameters, config, MODES)
        calls = []
        compare_callbacks = True

        def attend(index, queries, keys, values):
            calls.append(index)
            require(tuple(queries.shape) == (len(positions), 8, 8), "Noise query shape is incorrect")
            require(tuple(keys.shape) == tuple(values.shape) == (len(positions), 2, 8), "Noise KV shape is incorrect")
            require(queries.dtype == keys.dtype == values.dtype == dtype, "Noise QKV dtype is incorrect")
            output = padded_attention.attend(queries, keys, values, positions, request_ids,
                                             histories[index], model.attention_modes[index])
            if compare_callbacks:
                for name, actual, wanted in zip(("q", "k", "v", "attention"), (queries, keys, values, output), trace[index]):
                    compare(f"round{round_index}/noise/layer{index}/{name}", array(actual), wanted)
            return output

        output = model.forward(noise, position_tensor, attend)
        require(calls == list(range(6)), f"Forward callback order differs: {calls}")
        require(tuple(output.shape) == (len(positions), 32), "Output must be hidden states, not vocabulary logits")
        require(output.dtype == dtype, "Output dtype differs from model dtype")
        compare(f"round{round_index}/final", array(output), expected)
        compare(f"round{round_index}/ideal_fp64", array(output), ideal, enforce=False)
        compare(f"round{round_index}/external_head", array(output) @ head.T, expected @ head.T)
        require(not np.allclose(array(output) @ head.T, array(output) @ alternate_head.T), "Distinct external heads did not change logits")
        compare_callbacks = False
        calls.clear()
        repeated = model.forward(noise, position_tensor, attend)
        require(torch.equal(output, repeated), "Repeated forward with unchanged public inputs changed output")
        require(torch.equal(noise, noise_copy), "forward changed supplied embeddings")
        require(torch.equal(embedding, embedding_copy), "The caller's embedding table was changed")
        require(np.array_equal(position_tensor.numpy(), positions), "forward changed positions")

    for round_index, counts in enumerate([lengths] + extensions):
        context(np.array(counts), round_index)
        block(round_index)
    report["final_context_lengths"] = cursors.tolist()
    report["block_lengths"] = block_lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    report = {"tolerances": TOLERANCES, "device": "cpu", "cases": [], "failures": []}
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    try:
        from freetoken.speculative.dflash_model import DFlashModel, read_dflash_config
    except Exception as error:
        report["failures"].append({"case": "public_import", "error": f"{type(error).__name__}: {error}"})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["failures"]))
        return 1

    with tempfile.TemporaryDirectory(prefix="independent-dflash-math-") as temporary:
        directory = Path(temporary) / "neutral-checkpoint-directory"
        config, parameters, parameter_count = checkpoint(directory)
        report["fixture_config"] = config

        def case(name, operation):
            entry = {"name": name, "errors": []}
            report["cases"].append(entry)
            try:
                operation(entry)
                entry["passed"] = True
            except Exception as error:
                entry["passed"] = False
                entry["error"] = f"{type(error).__name__}: {error}"
                report["failures"].append({"case": name, "error": entry["error"]})
            print(f"{name}: {'PASS' if entry['passed'] else entry['error']}", flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")

        def config_contract(entry):
            loaded = read_dflash_config(directory)
            require(loaded.hidden_size == 32, "Public config reader changed hidden_size")
            model = DFlashModel(directory, dtype=torch.float32, device="cpu")
            require(model.hidden_size == 32 and model.block_size == 16 and model.mask_token_id == 46,
                    "Public model dimensions differ from checkpoint")
            require(list(model.target_layer_ids) == config["dflash_config"]["target_layer_ids"], "Target layer ordering changed")
            require(list(model.attention_modes) == MODES, "Causal/window/full attention modes differ")
            require(model.weight_bytes == parameter_count * 4, "weight_bytes differs from actual FP32 parameters")
            entry["weight_bytes"] = model.weight_bytes
            entry["input_embedding_scale"] = model.input_embedding_scale
            entry["output_multiplier"] = model.output_multiplier
            entry["final_logit_softcapping"] = model.final_logit_softcapping

        case("public_config_and_attributes", config_contract)
        scenarios = [
            ("single_repeated_extensions", [7], [9], [[1], [4]]),
            ("mixed_order_and_padding", [1, 3, 7, 19, 31], [1, 3, 5, 9, 16], [[1, 2, 3, 4, 5]]),
            ("c16_window_boundary", [4101] + list(range(1, 16)), [9] * 16, [[2] * 16]),
        ]
        for dtype in (torch.float32, torch.bfloat16):
            for name, lengths, blocks, extensions in scenarios:
                def evaluate(entry, dtype=dtype, lengths=lengths, blocks=blocks, extensions=extensions):
                    model = DFlashModel(directory, dtype=dtype, device="cpu")
                    expected_bytes = parameter_count * (4 if dtype == torch.float32 else 2)
                    require(model.weight_bytes == expected_bytes, "weight_bytes does not follow loaded dtype")
                    scenario(model, config, parameters, dtype, lengths, blocks, extensions, entry)
                case(f"{str(dtype).removeprefix('torch.')}/{name}", evaluate)

        def errors(entry):
            mutations = [
                ("missing_mask", lambda value: value["dflash_config"].pop("mask_token_id")),
                ("unsupported_activation", lambda value: value.update(hidden_act="relu")),
                ("unsupported_rope", lambda value: value["rope_parameters"].update(rope_type="yarn")),
                ("unsupported_attention", lambda value: value["layer_types"].__setitem__(0, "unsupported_attention")),
            ]
            for name, mutate in mutations:
                altered = copy.deepcopy(config)
                mutate(altered)
                path = Path(temporary) / name
                path.mkdir()
                (path / "config.json").write_text(json.dumps(altered))
                (path / "model.safetensors").write_bytes((directory / "model.safetensors").read_bytes())
                try:
                    DFlashModel(path, dtype=torch.float32, device="cpu")
                except Exception as error:
                    entry.setdefault("public_errors", {})[name] = f"{type(error).__name__}: {error}"
                else:
                    raise AssertionError(f"Invalid checkpoint accepted: {name}")
            weights = load_file(str(directory / "model.safetensors"))
            weights["fc.weight"] = weights["fc.weight"][:, :-1].contiguous()
            path = Path(temporary) / "wrong-weight-shape"
            path.mkdir()
            (path / "config.json").write_text(json.dumps(config))
            save_file(weights, str(path / "model.safetensors"))
            try:
                DFlashModel(path, dtype=torch.float32, device="cpu")
            except Exception as error:
                entry["public_errors"]["wrong_weight_shape"] = f"{type(error).__name__}: {error}"
            else:
                raise AssertionError("Mismatched fc.weight accepted")

        case("public_checkpoint_errors", errors)
    print(json.dumps({"failed": len(report["failures"]), "total": len(report["cases"])}))
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
