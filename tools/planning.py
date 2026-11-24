"""Fixed-loop visit and logical KV planning, using only the standard library.

No weights are loaded. The KV estimate excludes physical pages and all other
memory consumers. Local attention may have its own head configuration.
"""


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def text_config(config):
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    text = config.get("text_config", config)
    if not isinstance(text, dict):
        raise ValueError("text_config must be a JSON object")
    return text


def visit_plan(config):
    """Return visits in execution order, each with physical layer and pass id."""
    text = text_config(config)
    layers = integer(text.get("num_hidden_layers"), "num_hidden_layers")
    loop = text.get("loop_config")
    if not isinstance(loop, dict) or loop.get("is_varloop") is not False:
        raise ValueError("only explicit fixed-loop configurations are supported")
    start = integer(loop.get("core_start"), "core_start", 0)
    end = integer(loop.get("core_end"), "core_end")
    passes = integer(loop.get("num_passes"), "num_passes")
    if not 0 <= start < end <= layers:
        raise ValueError("core range must satisfy 0 <= start < end <= layers")
    local = text.get("local_layer_ids")
    if not isinstance(local, list) or any(type(x) is not int or not 0 <= x < layers for x in local):
        raise ValueError("local_layer_ids must be a list of valid physical layer IDs")
    if len(set(local)) != len(local):
        raise ValueError("local_layer_ids contains duplicates")
    local = set(local)
    pairs = [(i, 0) for i in range(start)]
    pairs += [(i, p) for p in range(passes) for i in range(start, end)]
    pairs += [(i, 0) for i in range(end, layers)]
    return [
        {"visit": n, "physical_layer": i, "pass": p,
         "attention": "local" if i in local else "global",
         "kv_owner": f"layers.{i}.attn.pass{p}",
         "conv_owner": f"layers.{i}.conv.pass{p}"}
        for n, (i, p) in enumerate(pairs)
    ]


def kv_estimate(config, *, tokens, tp=1, requests=1, element_bytes=2):
    """Estimate active logical KV elements assuming exact head sharding.

    Owner names in visit_plan are logical audit keys, not proposed runtime
    module prefixes. Real cache pages can consume substantially more memory.
    """
    text = text_config(config)
    tokens = integer(tokens, "tokens")
    tp = integer(tp, "tp")
    requests = integer(requests, "requests")
    element_bytes = integer(element_bytes, "element_bytes")
    window = integer(text.get("sliding_window_size"), "sliding_window_size")
    visits = visit_plan(config)
    shapes = {}
    for kind, prefix in (("global", ""), ("local", "swa_")):
        q = integer(text.get(prefix + "num_attention_heads", text.get("num_attention_heads")), prefix + "num_attention_heads")
        kv = integer(text.get(prefix + "num_key_value_heads", text.get("num_key_value_heads")), prefix + "num_key_value_heads")
        dim = integer(text.get(prefix + "head_dim", text.get("head_dim")), prefix + "head_dim")
        if q % tp or kv % tp:
            raise ValueError(f"TP={tp} must divide both {kind} Q heads and KV heads")
        shapes[kind] = (kv // tp, dim)
    totals = {"local": 0, "global": 0}
    for visit in visits:
        kind = visit["attention"]
        heads, dim = shapes[kind]
        retained = min(tokens, window) if kind == "local" else tokens
        totals[kind] += 2 * element_bytes * heads * dim * retained * requests
    return {
        "estimate_kind": "logical_active_kv_only_not_allocation",
        "tokens_per_request": tokens, "requests": requests, "tp": tp,
        "element_bytes": element_bytes,
        "physical_layers": text["num_hidden_layers"], "layer_visits": len(visits),
        "local_owners": sum(x["attention"] == "local" for x in visits),
        "global_owners": sum(x["attention"] == "global" for x in visits),
        "kv_owners": len(visits), "conv_owners": len(visits),
        "local_kv_bytes_per_rank": totals["local"],
        "global_kv_bytes_per_rank": totals["global"],
        "kv_bytes_per_rank": sum(totals.values()),
        "kv_gib_per_rank": sum(totals.values()) / 2**30,
        "excluded": ["weights", "conv_state", "message_bank", "page_alignment", "allocator_retention",
                     "metadata", "activations", "communication", "runtime", "graphs", "loading_peak"],
    }
