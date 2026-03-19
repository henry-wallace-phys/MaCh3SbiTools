"""
Tests for the two-level training progress display (logger.py).

Covers:
* create_progress returns the right type depending on show_progress
* TrainingProgress task IDs and initial state
* start_epoch resets the epoch bar and updates the fit description
* step_batch advances the epoch bar
* finish_epoch advances the fit bar and annotates both with loss strings
* The progress context manager actually starts/stops the Live display
* Disabled progress (nullcontext) path is safely no-op throughout the loop
"""

from contextlib import nullcontext

import pytest
from rich.progress import Progress

from mach3sbitools.utils.logger import TrainingProgress, create_progress

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tp() -> TrainingProgress:
    """A TrainingProgress built with show_progress=True."""
    return create_progress(
        total_epochs=100,
        steps_per_epoch=50,
        show_progress=True,
    )


# ---------------------------------------------------------------------------
# create_progress return type
# ---------------------------------------------------------------------------


class TestCreateProgressReturnType:
    def test_returns_training_progress_when_enabled(self):
        result = create_progress(total_epochs=10, steps_per_epoch=5, show_progress=True)
        assert isinstance(result, TrainingProgress)

    def test_returns_nullcontext_when_disabled(self):
        result = create_progress(
            total_epochs=10, steps_per_epoch=5, show_progress=False
        )
        assert isinstance(result, nullcontext)

    def test_nullcontext_is_usable_as_context_manager(self):
        result = create_progress(
            total_epochs=10, steps_per_epoch=5, show_progress=False
        )
        # Must not raise
        with result:
            pass


# ---------------------------------------------------------------------------
# TrainingProgress structure
# ---------------------------------------------------------------------------


class TestTrainingProgressStructure:
    def test_has_progress_instance(self, tp):
        assert isinstance(tp.progress, Progress)

    def test_has_two_tasks(self, tp):
        assert len(tp.progress.tasks) == 2

    def test_fit_task_total(self, tp):
        fit = tp.progress.tasks[tp.fit_task]
        assert fit.total == 100

    def test_epoch_task_total(self, tp):
        epoch = tp.progress.tasks[tp.epoch_task]
        assert epoch.total == 50

    def test_fit_task_starts_at_zero(self, tp):
        assert tp.progress.tasks[tp.fit_task].completed == 0

    def test_epoch_task_starts_at_zero(self, tp):
        assert tp.progress.tasks[tp.epoch_task].completed == 0

    def test_task_ids_are_distinct(self, tp):
        assert tp.fit_task != tp.epoch_task

    def test_extra_field_initialised_empty(self, tp):
        for task in tp.progress.tasks:
            assert task.fields["extra"] == ""


# ---------------------------------------------------------------------------
# start_epoch
# ---------------------------------------------------------------------------


class TestStartEpoch:
    def test_resets_epoch_bar_completed_to_zero(self, tp):
        # Advance the epoch bar first so there's something to reset
        tp.progress.advance(tp.epoch_task, 10)
        assert tp.progress.tasks[tp.epoch_task].completed == 10

        tp.start_epoch(epoch=2, total_epochs=100, n_steps=50)
        assert tp.progress.tasks[tp.epoch_task].completed == 0

    def test_updates_epoch_bar_total(self, tp):
        # Change n_steps mid-training (variable-length dataset)
        tp.start_epoch(epoch=1, total_epochs=100, n_steps=75)
        assert tp.progress.tasks[tp.epoch_task].total == 75

    def test_fit_description_contains_epoch_number(self, tp):
        tp.start_epoch(epoch=7, total_epochs=100, n_steps=50)
        desc = tp.progress.tasks[tp.fit_task].description
        assert "7" in desc

    def test_fit_description_contains_total_epochs(self, tp):
        tp.start_epoch(epoch=7, total_epochs=100, n_steps=50)
        desc = tp.progress.tasks[tp.fit_task].description
        assert "100" in desc

    def test_fit_bar_not_advanced_by_start_epoch(self, tp):
        tp.start_epoch(epoch=1, total_epochs=100, n_steps=50)
        assert tp.progress.tasks[tp.fit_task].completed == 0


# ---------------------------------------------------------------------------
# step_batch
# ---------------------------------------------------------------------------


class TestStepBatch:
    def test_advances_epoch_task_by_one(self, tp):
        before = tp.progress.tasks[tp.epoch_task].completed
        tp.step_batch()
        assert tp.progress.tasks[tp.epoch_task].completed == before + 1

    def test_does_not_advance_fit_task(self, tp):
        before = tp.progress.tasks[tp.fit_task].completed
        tp.step_batch()
        assert tp.progress.tasks[tp.fit_task].completed == before

    def test_multiple_steps_accumulate(self, tp):
        for _ in range(5):
            tp.step_batch()
        assert tp.progress.tasks[tp.epoch_task].completed == 5


# ---------------------------------------------------------------------------
# finish_epoch
# ---------------------------------------------------------------------------


class TestFinishEpoch:
    def test_advances_fit_task_by_one(self, tp):
        before = tp.progress.tasks[tp.fit_task].completed
        tp.finish_epoch(train_loss=1.5, val_loss=1.6)
        assert tp.progress.tasks[tp.fit_task].completed == before + 1

    def test_does_not_double_advance_epoch_task(self, tp):
        # finish_epoch should not advance the epoch bar — step_batch does that
        before = tp.progress.tasks[tp.epoch_task].completed
        tp.finish_epoch(train_loss=1.5, val_loss=1.6)
        assert tp.progress.tasks[tp.epoch_task].completed == before

    def test_fit_extra_contains_train_loss(self, tp):
        tp.finish_epoch(train_loss=1.2345, val_loss=9.9)
        extra = tp.progress.tasks[tp.fit_task].fields["extra"]
        assert "1.2345" in extra

    def test_fit_extra_contains_val_loss(self, tp):
        tp.finish_epoch(train_loss=9.9, val_loss=1.6789)
        extra = tp.progress.tasks[tp.fit_task].fields["extra"]
        assert "1.6789" in extra

    def test_epoch_extra_contains_losses(self, tp):
        tp.finish_epoch(train_loss=1.11, val_loss=2.22)
        extra = tp.progress.tasks[tp.epoch_task].fields["extra"]
        assert "1.1100" in extra
        assert "2.2200" in extra

    def test_multiple_epochs_accumulate_on_fit_bar(self, tp):
        for i in range(3):
            tp.finish_epoch(train_loss=float(i), val_loss=float(i))
        assert tp.progress.tasks[tp.fit_task].completed == 3


# ---------------------------------------------------------------------------
# Full simulated training loop
# ---------------------------------------------------------------------------


class TestSimulatedLoop:
    def test_fit_bar_equals_n_epochs_after_full_run(self):
        n_epochs = 5
        n_steps = 8
        tp = create_progress(
            total_epochs=n_epochs, steps_per_epoch=n_steps, show_progress=True
        )
        with tp.progress:
            for epoch in range(1, n_epochs + 1):
                tp.start_epoch(epoch, n_epochs, n_steps)
                for _ in range(n_steps):
                    tp.step_batch()
                tp.finish_epoch(train_loss=1.0, val_loss=1.1)

        assert tp.progress.tasks[tp.fit_task].completed == n_epochs

    def test_epoch_bar_equals_n_steps_after_last_epoch(self):
        n_epochs = 3
        n_steps = 6
        tp = create_progress(
            total_epochs=n_epochs, steps_per_epoch=n_steps, show_progress=True
        )
        with tp.progress:
            for epoch in range(1, n_epochs + 1):
                tp.start_epoch(epoch, n_epochs, n_steps)
                for _ in range(n_steps):
                    tp.step_batch()
                tp.finish_epoch(train_loss=0.5, val_loss=0.6)

        # After the last epoch the epoch bar should be at n_steps (not reset)
        assert tp.progress.tasks[tp.epoch_task].completed == n_steps

    def test_disabled_progress_loop_does_not_raise(self):
        ctx = create_progress(total_epochs=3, steps_per_epoch=4, show_progress=False)
        # The loop must work identically when progress is disabled
        with ctx:
            for epoch in range(1, 4):
                if isinstance(ctx, TrainingProgress):
                    ctx.start_epoch(epoch, 3, 4)
                for _ in range(4):
                    if isinstance(ctx, TrainingProgress):
                        ctx.step_batch()
                if isinstance(ctx, TrainingProgress):
                    ctx.finish_epoch(1.0, 1.1)
