"""Budget retained layered-prefill states separately from one step's workspace."""

from dataclasses import dataclass, field

import torch


@dataclass
class PrefillMemoryBudget:
    device: torch.device
    workspace_bytes: int = field(default=0, init=False)
    state_bytes_per_token: int = field(default=0, init=False)
    measured_rows: int = field(default=0, init=False)

    def start(self) -> int:
        torch.cuda.reset_peak_memory_stats(self.device)
        return torch.cuda.memory_allocated(self.device)

    def record(self, before: int, rows: int, *, first_stage: bool) -> None:
        after = torch.cuda.memory_allocated(self.device)
        peak = torch.cuda.max_memory_allocated(self.device)
        # Include the current tile's new state in the workspace reserve. Later
        # groups may replace states, but only the first group adds retained rows.
        self.workspace_bytes = max(self.workspace_bytes, peak - min(before, after))
        if first_stage:
            self.state_bytes_per_token = max(
                self.state_bytes_per_token, (after - before + rows - 1) // rows
            )
        self.measured_rows = max(self.measured_rows, rows)

    def token_budget(self, tile_tokens: int, max_tokens: int) -> int:
        # Bootstrap with useful work, not an extra model probe. A shorter request
        # does not establish the workspace needed by a full physical tile.
        if self.measured_rows < tile_tokens or not self.state_bytes_per_token:
            return min(tile_tokens, max_tokens)
        free, _ = torch.cuda.mem_get_info(self.device)
        available = (
            free + torch.cuda.memory_reserved(self.device)
            - torch.cuda.memory_allocated(self.device)
        )
        rows = (available - self.workspace_bytes) // self.state_bytes_per_token
        # The configured single tile must fit, just as for ordinary chunked prefill.
        rows = max(tile_tokens, rows // tile_tokens * tile_tokens)
        return min(rows, max_tokens)
