"""FSDP worker wrappers that load ARFT policy-loss registrations per Ray actor."""

from __future__ import annotations

from verl.single_controller.base.decorator import make_nd_compute_dataproto_dispatch_fn, register
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker as _VerlAsyncActorRolloutRefWorker


class AsyncActorRolloutRefWorker(_VerlAsyncActorRolloutRefWorker):
    """Upstream FSDP worker with ARFT-only policy-loss registration."""

    def __init__(self, *args, **kwargs):
        # Ray actors are independent processes; registration in the driver alone
        # would not make the custom loss mode visible here.
        import arft.policy_losses  # noqa: F401

        super().__init__(*args, **kwargs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data):
        import arft.policy_losses

        arft.policy_losses.set_global_step(data.non_tensor_batch.get("global_step"))
        return super().update_actor(data)
