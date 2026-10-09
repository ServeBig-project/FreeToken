import torch


class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor, rows=None) -> None:
        self._max_running_reqs = max_running_reqs
        self._rows = rows  # per-row storage that is mapped while a request holds the row
        self._free_slots = self._all_rows()
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    def _all_rows(self) -> list[int]:
        # Rows with mapped per-row storage are handed out lowest first, so the rows in use
        # pack from the start of their storage like every other unit of a shared runtime.
        rows = list(range(self._max_running_reqs))
        return rows[::-1] if self._rows is not None else rows

    @property
    def available_size(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        return self._free_slots.pop()

    def peek(self) -> int | None:
        """The row the next allocation takes (not taken yet)."""
        return self._free_slots[-1] if self._free_slots else None

    def take(self, slot: int) -> None:
        """Record ``peek``'s row as allocated once its per-row storage is held."""
        assert self._free_slots.pop() == slot

    @property
    def row_units(self):
        """Per-row storage mapped while a request holds the row (ReplaySSM records)."""
        return self._rows.units if self._rows is not None else None

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)
        if self._rows is not None:
            self._rows.unbind(slot)

    def rebuild(self, page_table: torch.Tensor, rows=None) -> None:
        """Re-point the page table, reallocate the token pool, and free all slots.

        Idle-only: all request slots are expected to be free at rebuild time.
        """
        self.page_table = page_table
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)
        self._rows = rows if rows is not None else self._rows
        self._free_slots = self._all_rows()
