# copy/pasted from pytorch lightning
# https://github.com/Lightning-AI/lightning/blob/0d52f4577310b5a1624bed4d23d49e37fb05af9e/src/lightning_fabric/utilities/seed.py
# and
# https://github.com/Lightning-AI/lightning/blob/98f7696d1681974d34fad59c03b4b58d9524ed13/src/pytorch_lightning/utilities/seed.py

# Copyright The Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
from typing import Generator, Dict, Any

import torch
import numpy as np
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state


def _collect_rng_states(include_cuda: bool = True) -> Dict[str, Any]:
    """Collect the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python."""
    states = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": python_get_rng_state(),
    }
    if include_cuda:
        try:
            states["torch.cuda"] = torch.cuda.get_rng_state_all()
        except RuntimeError:
            # CUDA initialization failure.
            pass
    return states


def _set_rng_states(rng_state_dict: Dict[str, Any]) -> None:
    """Set the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python in the current
    process."""
    torch.set_rng_state(rng_state_dict["torch"])
    # torch.cuda rng_state is only included since v1.8.
    if "torch.cuda" in rng_state_dict:
        torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])
    np.random.set_state(rng_state_dict["numpy"])
    version, state, gauss = rng_state_dict["python"]
    python_set_rng_state((version, tuple(state), gauss))


@contextmanager
def isolate_rng(include_cuda: bool = True) -> Generator[None, None, None]:
    """A context manager that resets the global random state on exit to what it was before entering.
    It supports isolating the states for PyTorch, Numpy, and Python built-in random number generators.
    Args:
        include_cuda: Whether to allow this function to also control the `torch.cuda` random number generator.
            Set this to ``False`` when using the function in a forked process where CUDA re-initialization is
            prohibited.
    Example:
        >>> import torch
        >>> torch.manual_seed(1)  # doctest: +ELLIPSIS
        <torch._C.Generator object at ...>
        >>> with isolate_rng():
        ...     [torch.rand(1) for _ in range(3)]
        [tensor([0.7576]), tensor([0.2793]), tensor([0.4031])]
        >>> torch.rand(1)
        tensor([0.7576])
    """
    states = _collect_rng_states(include_cuda)
    yield
    _set_rng_states(states)
