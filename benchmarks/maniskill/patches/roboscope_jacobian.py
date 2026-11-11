"""Opt-in, fixed-shape inference Jacobian graphs for two pinned robot chains.

The FK implementation is adapted from pytorch-kinematics 0.7.4, chain.py and
transforms/rotation_conversions.py. The upstream calc_jacobian, frame transforms,
rotation arithmetic and matrix multiplication order are retained.

Copyright (c) 2023 University of Michigan ARM Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from pytorch_kinematics.jacobian import calc_jacobian
from pytorch_kinematics.transforms import Transform3d
from pytorch_kinematics.transforms.rotation_conversions import axis_and_angle_to_matrix_33

PATCH_VERSION = "jacobian-cuda-graphs-v2"
SWITCH = "ROBOSCOPE_MANISKILL_JACOBIAN_GRAPHS"
# Exactness probes cover these end links and batch widths. Other paths stay upstream.
SUPPORTED_CHAINS = {
    ("524c6986dbbabe244e05801f262862205e880834d01459b06ee273016d0579ba", "panda_hand_tcp"):
        ("panda", 7, 10),
    ("1e8a73f270804dbcc32603d68a8fb5d34f57d109b30cc325ad89b36eecbb8506", "ee_gripper_link"):
        ("widowx_bridge", 6, 1),
}
# Kinematics is a pure function of joint values and pinned URDF geometry. Keep
# at most two graph entries per Worker, independent of simulator/episode lifetime.
_GRAPH_CACHE: OrderedDict[tuple[Any, ...], _GraphEntry] = OrderedDict()


def _record(event: str, **fields: Any) -> None:
    # One record per chain setup/capture, never a per-control log.
    print("ROBOSCOPE_JACOBIAN_GRAPH " + json.dumps({
        "event": event, "patch_version": PATCH_VERSION, "pid": os.getpid(),
        "switch": os.environ.get(SWITCH), **fields,
    }, sort_keys=True), flush=True)


def _homogeneous_row(axis: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.zeros(3, dtype=axis.dtype, device=axis.device),
                      torch.ones(1, dtype=axis.dtype, device=axis.device)))


def _rotation44(axis: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    rot = axis_and_angle_to_matrix_33(axis, theta)
    batch_shape = axis.shape[:-1]
    mat44 = torch.cat((rot, torch.zeros(*batch_shape, 3, 1, dtype=axis.dtype, device=axis.device)), -1)
    return torch.cat((mat44, _homogeneous_row(axis).expand(*batch_shape, 1, 4)), -2)


def _prismatic44(axis: torch.Tensor, distance: torch.Tensor) -> torch.Tensor:
    batch_axes = axis.shape[:-1]
    mat33 = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(*batch_axes, 3, 3)
    pos = axis * distance.unsqueeze(-1)
    mat44 = torch.cat((mat33, pos.unsqueeze(-1)), -1)
    return torch.cat((mat44, _homogeneous_row(axis).expand(*batch_axes, 1, 4)), -2)


class _CaptureChain:
    """Read original chain fields without inserting this shadow into upstream LRU caches."""

    def __init__(self, chain: Any) -> None:
        self._chain = chain
        self.parents = [tuple(int(i) for i in row.cpu().tolist()) for row in chain.parents_indices]
        self.joint_indices = tuple(int(i) for i in chain.joint_indices.cpu().tolist())
        self.joint_types = tuple(int(i) for i in chain.joint_type_indices.cpu().tolist())
        self.end_index = chain.frame_to_idx[chain._serial_frames[-1].name]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chain, name)

    def forward_kinematics(self, th: torch.Tensor) -> Transform3d:
        # Equivalent to upstream SerialChain end_only=True and Chain.forward_kinematics.
        # Topology scalars are pre-extracted; identities/constants start on the target device.
        th = self._chain.ensure_tensor(th)
        th = torch.atleast_2d(th)
        batch = th.shape[0]
        axes_expanded = self.axes.unsqueeze(0).repeat(batch, 1, 1)
        revolute = _rotation44(axes_expanded, th)
        prismatic = _prismatic44(axes_expanded, th)
        frame_transforms = {}
        frame_transform = torch.eye(4, dtype=th.dtype, device=th.device).unsqueeze(0).repeat(batch, 1, 1)
        for chain_index in self.parents[self.end_index]:
            if chain_index in frame_transforms:
                frame_transform = frame_transforms[chain_index]
            else:
                link_offset = self.link_offsets[chain_index]
                if link_offset is not None:
                    frame_transform = frame_transform @ link_offset
                joint_offset = self.joint_offsets[chain_index]
                if joint_offset is not None:
                    frame_transform = frame_transform @ joint_offset
                joint_index = self.joint_indices[chain_index]
                joint_type = self.joint_types[chain_index]
                if joint_type == 0:
                    pass
                elif joint_type == 1:
                    frame_transform = frame_transform @ revolute[:, joint_index]
                elif joint_type == 2:
                    frame_transform = frame_transform @ prismatic[:, joint_index]
        frame_transforms[self.end_index] = frame_transform
        # The original end-only path constructs exactly this one frame Transform3d.
        return Transform3d(matrix=frame_transforms[self.end_index])


class _GraphEntry:
    """Own static geometry and buffers; never reference a simulator or proxy."""

    def __init__(self, shadow: _CaptureChain, value: torch.Tensor) -> None:
        # Captured kernels retain addresses, so their immutable geometry must live
        # as long as the graph. The source chain was built only from the URDF.
        self.geometry = shadow
        self.input = torch.empty_like(value)
        self.input.copy_(value)
        stream = torch.cuda.Stream(device=value.device)
        current = torch.cuda.current_stream(value.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                calc_jacobian(shadow, self.input, ret_eef_pose=True)
        current.wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.no_grad():
            self.jacobian, self.pose = calc_jacobian(shadow, self.input, ret_eef_pose=True)

    def replay(self, value: torch.Tensor, ret_eef_pose: bool) -> Any:
        self.input.copy_(value)
        self.graph.replay()
        # Static graph outputs are overwritten next call; callers receive independent tensors.
        jacobian = self.jacobian.clone()
        return (jacobian, self.pose.clone()) if ret_eef_pose else jacobian


class _GraphedChain:
    def __init__(self, chain: Any, scope: tuple[str, int, int], geometry_key: tuple[str, str]) -> None:
        self._chain = chain
        self._shadow = _CaptureChain(chain)
        self._scope = scope
        self._geometry_key = geometry_key

    def __getattr__(self, name: str) -> Any:
        return getattr(self._chain, name)

    def jacobian(self, th: Any, locations: Any = None, **kwargs: Any) -> Any:
        ret_eef_pose = kwargs.get("ret_eef_pose", False)
        if (not isinstance(th, torch.Tensor) or th.device.type != "cuda" or th.requires_grad
                or th.dtype != torch.float32 or th.dtype != self._chain.dtype
                or th.device != self._chain.axes.device
                or torch.is_autocast_enabled()
                or th.ndim != 2 or tuple(th.shape) != (self._scope[2], self._scope[1])
                or not th.is_contiguous() or locations is not None
                or not isinstance(ret_eef_pose, bool) or set(kwargs) - {"ret_eef_pose"}):
            return self._chain.jacobian(th, locations=locations, **kwargs)
        key = (self._geometry_key, th.device.type, th.device.index, th.dtype, tuple(th.shape),
               torch.backends.cuda.matmul.allow_tf32,
               torch.get_float32_matmul_precision(), torch.is_inference_mode_enabled(),
               torch.cuda.current_stream(th.device).cuda_stream)
        with torch.cuda.device(th.device):
            if torch.cuda.is_current_stream_capturing():
                return self._chain.jacobian(th, locations=locations, **kwargs)
            entry = _GRAPH_CACHE.get(key)
            if entry is None:
                start = time.perf_counter()
                allocated = torch.cuda.memory_allocated(th.device)
                reserved = torch.cuda.memory_reserved(th.device)
                try:
                    entry = _GraphEntry(self._shadow, th)
                except Exception as error:
                    # Failed CUDA capture can invalidate the context; silently falling back is unsafe.
                    message = f"{PATCH_VERSION}: Jacobian capture failed for {self._scope}"
                    raise RuntimeError(message) from error
                _GRAPH_CACHE[key] = entry
                # Every entry owns its static geometry; old environments can close.
                if len(_GRAPH_CACHE) > 2:
                    _GRAPH_CACHE.popitem(last=False)
                _record("capture", chain=self._scope[0], shape=list(th.shape), dtype=str(th.dtype),
                        device=str(th.device), setup_cpu_wall_s=time.perf_counter()-start,
                        allocated_delta_bytes=torch.cuda.memory_allocated(th.device)-allocated,
                        reserved_delta_bytes=torch.cuda.memory_reserved(th.device)-reserved)
            else:
                _GRAPH_CACHE.move_to_end(key)
            return entry.replay(th, ret_eef_pose)


def maybe_graph_chain(chain: Any, *, urdf_path: str, end_link_name: str) -> Any:
    """Default-off, per-Kinematics scope; all unproven robots and call paths remain upstream."""
    requested = os.environ.get(SWITCH, "0")
    if requested not in {"0", "1"}:
        raise ValueError(f"{SWITCH} must be 0 or 1")
    digest = hashlib.sha256(Path(urdf_path).read_bytes()).hexdigest()
    scope = SUPPORTED_CHAINS.get((digest, end_link_name))
    enabled = requested == "1" and scope is not None
    _record("configuration", requested=requested, eligible=scope is not None, enabled=enabled,
            urdf_sha256=digest, end_link=end_link_name, chain=scope[0] if scope else None,
            supported_batch=scope[2] if scope else None)
    return _GraphedChain(chain, scope, (digest, end_link_name)) if enabled else chain
