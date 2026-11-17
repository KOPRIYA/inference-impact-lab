"""Exact-token cache matching; cache identity binds tenant, model and policy."""
import copy
import hashlib
import json


def common_prefix(sequences):
    if not sequences:
        return []
    shared = []
    for values in zip(*sequences):
        if len(set(values)) != 1:
            break
        shared.append(values[0])
    # Keep at least one token for generation even for identical complete prompts.
    return shared[:max(0, min(len(s) for s in sequences) - 1)]


class PrefixSnapshot:
    def __init__(self, tokens, state, model_identity, namespace, policy_hash):
        if not tokens:
            raise ValueError("a prefix must be nonempty")
        self.tokens = tuple(tokens)
        self.state = state
        self.identity = (model_identity, namespace, policy_hash)
        self.key = hashlib.sha256(json.dumps([self.identity, self.tokens]).encode()).hexdigest()

    def checkout(self, tokens, model_identity, namespace, policy_hash):
        if self.identity != (model_identity, namespace, policy_hash):
            raise ValueError("cache identity mismatch")
        n = len(self.tokens)
        if tuple(tokens[:n]) != self.tokens or len(tokens) <= n:
            raise ValueError("prompt does not extend cached prefix")
        # The snapshot must never acquire another request's generated tokens.
        return list(tokens[n:]), copy.deepcopy(self.state)
